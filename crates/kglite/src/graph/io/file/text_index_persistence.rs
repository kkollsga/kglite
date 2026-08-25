//! BM25 text-index `.kgl` section — the lexical twin of
//! [`vector_persistence`](super::vector_persistence).

use super::{codec_deser, codec_ser};
use crate::graph::algorithms::text_index::{Posting, TextIndex};
use crate::graph::index_freshness::IndexFreshness;
use crate::graph::schema::DirGraph;
use crate::graph::text_indexes::{self, TextIndexRead};
use crate::serde_codec;
use serde::{Deserialize, Serialize};
use std::io;

// ─── BM25 text-index section (0.16.10) ────────────────────────────────────
//
// A self-describing, *skippable* `.kgl` sub-section carrying built text
// indexes, framed exactly like the vector section beside it and for the same
// reason: the index is a rebuildable cache, never a correctness dependency, so
// any version mismatch or incoherent payload is silently dropped. The graph
// loads without it and the user rebuilds.
//
//   [0..8]   magic = b"KGLTIDX1"
//   [8..12]  format_version: u32 LE
//   [12..]   codec payload for Vec<PersistedTextIndex>
//
// The payload version is this section's own, independent of the core data
// version and of the vector section's: the two index formats evolve on their
// own schedules, and neither move is a `.kgl` format break.
//
// **What is written is the logical index, not its memory layout.**
// `TextIndex` derives `Serialize`, so dumping the struct would have been fewer
// lines, and it does round-trip into a valid index — that route was measured
// before this one was chosen, and rejected on two counts. It is *not
// canonical*: both of its maps are hash maps, so their iteration order is a
// function of insertion history and two graphs holding the same corpus
// serialize to different bytes, which the rest of this writer goes to some
// length to avoid. And it is *fat*: the dictionary is stored in both
// directions (`ids` and `names` share one `Arc<str>` per term, a sharing serde
// cannot express), so every term's bytes are written twice and read back into
// two allocations. On the six-document fixture in the tests the struct dump is
// 351 bytes and order-dependent; the form below is 191 and byte-identical
// across equivalent corpora.
//
// What is written instead is `iter_terms`' id-independent view — each term
// once, by name, in sorted order, with its postings — and the index is rebuilt
// from it on load (`TextIndex::from_terms`). The forward view is the transpose
// of the postings and is derived rather than carried; the free list a churned
// index accumulates does not survive, because term ids are internal and the
// round-trip compacts them.
pub(super) const TEXT_INDEX_MAGIC: &[u8; 8] = b"KGLTIDX1";
const TEXT_INDEX_FORMAT_VERSION: u32 = 1;

/// One index held open for the duration of an encode. The read guard is what
/// keeps a concurrent catch-up from renumbering the dictionary mid-write.
struct HeldIndex<'a> {
    node_type: &'a str,
    property: &'a str,
    resolved_field: &'a str,
    skipped: usize,
    watermark: u32,
    limit: usize,
    dirty: Vec<u32>,
    guard: TextIndexRead<'a>,
}

/// One index as written — borrowed so a save does not clone a corpus-sized
/// postings map. Postcard encodes struct fields positionally, so this and
/// [`PersistedTextIndex`] are the same bytes.
#[derive(Serialize)]
struct PersistedTextIndexRef<'a> {
    node_type: &'a str,
    property: &'a str,
    resolved_field: &'a str,
    skipped: usize,
    watermark: u32,
    limit: usize,
    dirty: Vec<u32>,
    terms: Vec<(&'a str, &'a [Posting])>,
    empty_docs: Vec<u32>,
}

/// One index's persisted form: the corpus, what the index has yet to cover,
/// and the build-time facts a refresh cannot restate.
#[derive(Serialize, Deserialize)]
struct PersistedTextIndex {
    node_type: String,
    property: String,
    /// The alias-resolved column the *build* read, so a refresh after a reload
    /// cannot silently repoint at another one.
    resolved_field: String,
    /// Nodes of the type that yielded no document at build time. Carried, not
    /// recomputed — see `attach_persisted_text_index`.
    skipped: usize,
    /// Node slots the index covers.
    watermark: u32,
    /// The inline-refresh ceiling this index was built with.
    limit: usize,
    /// Slots changed in place since the last catch-up. Sorted on write so
    /// equivalent graphs serialize byte-identically.
    dirty: Vec<u32>,
    /// `(term, postings)`, sorted by term — the whole corpus, id-independent.
    terms: Vec<(String, Vec<Posting>)>,
    /// Slots whose document holds no term. They appear in no posting list, and
    /// losing them would change BM25's `N` and therefore every score.
    empty_docs: Vec<u32>,
}

/// Encode every built text index into a self-describing payload. Returns
/// `None` when the graph has none (the section is then omitted entirely).
///
/// Reads each index as it stands and never refreshes it: saving must record
/// what is actually there, not perform a catch-up as a side effect of
/// `save_graph`.
pub(super) fn encode_text_indexes(graph: &DirGraph) -> io::Result<Option<Vec<u8>>> {
    // Already sorted by (node_type, property) — the one enumeration order.
    let stores = text_indexes::list_text_indexes(graph);
    if stores.is_empty() {
        return Ok(None);
    }
    let held: Vec<HeldIndex<'_>> = stores
        .into_iter()
        .map(|(node_type, property, store)| {
            let (watermark, limit, dirty) = store.freshness_state().persisted_parts();
            HeldIndex {
                node_type,
                property,
                resolved_field: store.resolved_field(),
                skipped: store.skipped(),
                watermark,
                limit,
                dirty,
                guard: store.read(),
            }
        })
        .collect();
    let entries: Vec<PersistedTextIndexRef<'_>> = held
        .iter()
        .map(|held| {
            let index = held.guard.index();
            // Sorted by term rather than left in id order: ids depend on the
            // order documents arrived, so two equivalent corpora would
            // otherwise write different bytes.
            let mut terms: Vec<(&str, &[Posting])> = index.iter_terms().collect();
            terms.sort_unstable_by_key(|(term, _)| *term);
            let mut empty_docs: Vec<u32> = index
                .doc_slots()
                .filter(|slot| index.doc_len(*slot) == Some(0))
                .collect();
            empty_docs.sort_unstable();
            PersistedTextIndexRef {
                node_type: held.node_type,
                property: held.property,
                resolved_field: held.resolved_field,
                skipped: held.skipped,
                watermark: held.watermark,
                limit: held.limit,
                dirty: held.dirty.clone(),
                terms,
                empty_docs,
            }
        })
        .collect();
    let body = codec_ser(serde_codec::CodecVersion::PostcardV1, &entries)?;
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(TEXT_INDEX_MAGIC);
    payload.extend_from_slice(&TEXT_INDEX_FORMAT_VERSION.to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(Some(payload))
}

/// Decode the text-index section and attach each index to the loaded graph.
///
/// Best-effort: an unrecognised magic, an unknown format version, a codec
/// error, a payload whose corpus does not describe a coherent index, or a type
/// this file no longer carries all result in that index being silently
/// skipped — never a load failure.
pub(super) fn decode_text_indexes(payload: &[u8], graph: &mut DirGraph) {
    if payload.len() < 12 || &payload[..8] != TEXT_INDEX_MAGIC {
        return;
    }
    let ver = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    if ver != TEXT_INDEX_FORMAT_VERSION {
        return; // rebuildable cache: skip anything this build cannot read
    }
    let codec = serde_codec::CodecVersion::PostcardV1;
    let entries: Vec<PersistedTextIndex> =
        match codec_deser(codec, &payload[12..], (payload.len() - 12) as u64) {
            Ok(entries) => entries,
            Err(_) => return,
        };
    for entry in entries {
        // A dirty slot at or above the watermark is not a state this tracker
        // can produce (`note_changed` ignores those — the gap already walks
        // them), and a refresh would then walk it twice.
        if entry.dirty.iter().any(|slot| *slot >= entry.watermark) {
            continue;
        }
        if !graph.has_node_type(&entry.node_type) {
            continue;
        }
        let index = TextIndex::from_terms(entry.terms, &entry.empty_docs);
        if index.validate().is_err() {
            continue;
        }
        let freshness = IndexFreshness::restored(entry.watermark, entry.limit, &entry.dirty);
        text_indexes::attach_persisted_text_index(
            graph,
            &entry.node_type,
            &entry.property,
            index,
            freshness,
            entry.resolved_field,
            entry.skipped,
        );
    }
}

#[cfg(test)]
#[path = "text_index_persistence_tests.rs"]
mod tests;
