//! How much memory loading a `.kgl` will cost, answered from the file's
//! metadata head alone — and the ceiling that refuses a load whose answer is
//! too large.
//!
//! # What it reads
//!
//! Only the JSON metadata block at the head of a v5/v6 container: the u32 at
//! bytes `[9..13]` gives its length, the block itself starts at 13. That block
//! is 0.01%–0.35% of the file on the four measured fixtures, so an estimate
//! costs one short read and no decompression at all.
//!
//! # Which number it estimates
//!
//! **Physical footprint, not RSS.** `phys_footprint` is the metric macOS
//! jetsam judges, and RSS overstates it by up to 3.6× on this load path —
//! columns of 256 KB or more are written to a spill file and mmap'd, so their
//! pages are clean and file-backed. RSS is also allocator-dependent (a 2.3×
//! swing between libmalloc's default and `MallocSpaceEfficient=1` on the same
//! bytes), while footprint agreed to 7% across two allocators on the same
//! fixture. A ceiling calibrated on RSS would refuse loads that fit
//! comfortably. See `dev-docs/bench/results/load-rss-2026-08-29.md` §0.
//!
//! **Touched bytes, not allocated capacity.** The decompression path leaves
//! `Vec` capacity at 1.2×–1.9× of length, and those slack pages are never
//! written, so they are never resident and never counted. Modelling capacity
//! over-predicts by ~40% and measures nothing (§7 of the same results).
//!
//! **The load-settled plateau, plus the load transient.** A 500k-row graph
//! grows a *third* plateau on its first point lookup, when the lazy per-type
//! `id_indices` build: +30.7% settled footprint, measured (§3). That term is
//! **not** included here — this estimates what a load costs, not what a
//! queried graph costs.
//!
//! # Which terms are modelled and which are guessed
//!
//! | term | status |
//! |---|---|
//! | [`LoadMemoryEstimate::index_rebuild_bytes`] | **modelled** — the file declares exactly which indexes exist and over which type, and the type's row count is in the same metadata |
//! | [`LoadMemoryEstimate::section_heap_bytes`] | **heuristic** — per-row and per-cell constants; measured 0.56×–1.30× of actual across three fixtures |
//! | [`LoadMemoryEstimate::transient_peak_bytes`] | **heuristic** — one decompression ratio against a measured 2.4×–6.1× band |
//!
//! Every constant below carries the measurement it came from. When the load
//! path changes shape, the calibration test at the bottom of this file is what
//! notices.

use std::io;

use super::{FileMetadata, PortableColumnSection};

// ─── Section-heap constants (HEURISTIC) ──────────────────────────────────────
//
// Derivation: fit against the settled footprint of three fixtures measured at
// release profile under the tight allocator (`dev-docs/bench/results/
// load-rss-2026-08-29.md` §1, `cli / spaceeff` rows), with the index term
// subtracted where the fixture declares indexes:
//
//   fixture          rows      cells    measured   this model   ratio
//   sodir          546,850  7,128,312   168.3 MB     218.9 MB   1.30×
//   codebase        23,142    663,202    27.8 MB      15.6 MB   0.56×
//   *_500k         500,000  2,500,000    86.6 MB     108.0 MB   1.25×
//
// **That 0.56×–1.30× band is the error bar, and it is not noise** — it is the
// spread between corpora. `codebase` holds long strings (code signatures,
// doc text) at 1,201 B/node; the 500k fixture holds short generated ones at
// 173 B/node. No linear function of (rows, cells) fits both, and the file's
// metadata does not record string lengths, so a tighter model would need to
// read the payload — which is exactly the cost this function exists to avoid.

/// Per node, independent of its properties: the petgraph slot, `NodeData`'s
/// header, its interned type key, and its id/title strings (which live in node
/// slots rather than in any column section, so no cell below counts them).
const NODE_ROW_BYTES: u64 = 64;

/// Per string/mixed cell: the `Option<String>`-shaped slot plus the mean
/// content that survives into the store. Strings that reach the 256 KB
/// per-column spill threshold are mmap'd and drop out of footprint entirely,
/// which is part of why the band above is as wide as it is.
const STRING_CELL_BYTES: u64 = 40;

/// Per int64/float64/date/uniqueid cell: an 8-byte payload in a nullable slot.
const NUMERIC_CELL_BYTES: u64 = 16;

/// Per bool cell.
const BOOL_CELL_BYTES: u64 = 8;

// ─── Index-rebuild constants (MODELLED) ──────────────────────────────────────
//
// Derivation: `indexed_500k` minus `noindex_500k` — identical nodes, edges and
// properties, differing only in four declared index structures over 500,000
// rows — measured the index term at **64.1 MB** (cli/spaceeff footprint) and
// **79.1 MB** (wheel/mimalloc footprint); §4 of the results doc, which also
// records why the two RSS readings of the same pair are instrument artifacts
// and must not be used. This model predicts 73.5 MB for that file: 1.15× the
// low reading, 0.93× the high one.
//
// The structures the constants describe:
//
// * a unique constraint is `HashMap<CompositeValue, NodeIndex>` with **one
//   entry per row** — distinct by definition, which is what makes this the one
//   index family whose key count needs no assumption;
// * an equality/range index is `LayeredIndex<Value>`, i.e.
//   `HashMap<Value, Vec<NodeIndex>>`: one posting slot per row plus one key per
//   *distinct* value, and the file does not record cardinality.

/// Per unique-constraint row: the map slot for `(CompositeValue, NodeIndex)`
/// plus hashbrown's occupancy headroom, before the key's own payload.
const UNIQUE_ENTRY_BYTES: u64 = 48;

/// Per indexed row in an equality/range/composite index: the `NodeIndex` in its
/// posting vector, plus the bytes a growth memcpy touches on the way there.
const POSTING_ENTRY_BYTES: u64 = 8;

/// Assumed distinct-value density of an equality/range/composite index:
/// one distinct key per this many rows.
///
/// A guess, and deliberately a conservative one: an index whose values were all
/// distinct is a *unique constraint*, and the shapes people actually index
/// (category, status, region, tag, kind) sit far below 1-in-8 — the calibration
/// fixture's are 1-in-2,500 and 1-in-12,500. Over-assuming keys makes the
/// estimate larger, which for a ceiling is the safe direction.
const KEY_DENSITY_DIVISOR: u64 = 8;

/// The `Vec` header a `CompositeValue` key carries on top of its values.
const COMPOSITE_KEY_VEC_BYTES: u64 = 24;

/// Per indexed value, by the column's declared type tag: what one key's payload
/// costs inside a `Value`.
const STRING_VALUE_BYTES: u64 = 48;
const SCALAR_VALUE_BYTES: u64 = 16;

// ─── Transient constants (HEURISTIC) ─────────────────────────────────────────

/// Decompressed size of a section, as a multiple of its compressed size,
/// ×2 (integer arithmetic: the ratio is 4.5).
///
/// Measured on four real sections: sodir's largest column 3.07×, its topology
/// 4.00×; `indexed_500k`'s largest column 5.87×, its topology 4.70× — inside
/// the 2.4×–6.1× band §7 reports across every section it probed. The midpoint
/// predicts those two column sections at 1.47× and 0.77× of their measured
/// decompressed size.
const DECOMPRESS_RATIO_NUMERATOR: u64 = 9;
const DECOMPRESS_RATIO_DENOMINATOR: u64 = 2;

/// What loading a `.kgl` is estimated to cost, in bytes of **physical
/// footprint**, broken into the terms it is made of rather than collapsed into
/// one number — because the terms have different accuracies and different
/// remedies, and a caller that is over a ceiling needs to know which one is
/// large.
///
/// Produced by [`estimate_load_memory`] / [`estimate_load_memory_bytes`] from
/// the file's metadata head. Read the module documentation before acting on
/// one: it states which metric this is (footprint, not RSS), which plateau
/// (load-settled, not post-first-lookup), and which terms are modelled versus
/// guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadMemoryEstimate {
    /// **Modelled.** What rebuilding the file's declared indexes costs, summed
    /// over every property, range, composite and unique declaration the
    /// metadata carries, each against its own node type's row count.
    ///
    /// Zero for a file that declares no index — and for a load that passes
    /// [`LoadOptions::defer_index_rebuild`](super::LoadOptions), which is
    /// exactly the term the deferral removes.
    pub index_rebuild_bytes: u64,
    /// **Heuristic.** What the decoded graph itself costs once settled:
    /// topology plus every column section's cells. Accurate to 0.56×–1.30× on
    /// the calibration corpora (see the module doc's table).
    pub section_heap_bytes: u64,
    /// **Heuristic.** The largest single decompression buffer the load holds
    /// live — the load's peak sits one section above its settled cost, and that
    /// section is a *column* section on every fixture measured, not the
    /// topology.
    pub transient_peak_bytes: u64,
    /// Total node rows the file declares, across every column section. Reported
    /// because it is the scale the two modelled terms are proportional to, and
    /// because a caller comparing two files wants it.
    pub node_rows: u64,
    /// How many index declarations [`Self::index_rebuild_bytes`] summed.
    pub declared_indexes: u32,
}

impl LoadMemoryEstimate {
    /// What the graph is estimated to cost once the load returns —
    /// [`Self::section_heap_bytes`] + [`Self::index_rebuild_bytes`].
    ///
    /// Excludes the lazy `id_indices` plateau a first point lookup builds
    /// (+30.7% measured on a 500k-row type); see the module doc.
    pub fn total_settled_bytes(&self) -> u64 {
        self.section_heap_bytes
            .saturating_add(self.index_rebuild_bytes)
    }

    /// The high-water mark *during* the load — [`Self::total_settled_bytes`]
    /// plus the one decompression buffer held live over it.
    ///
    /// This is the number a ceiling compares against, because a process dies at
    /// its peak rather than at its resting size.
    pub fn total_peak_bytes(&self) -> u64 {
        self.total_settled_bytes()
            .saturating_add(self.transient_peak_bytes)
    }

    /// [`Self::total_peak_bytes`] under a given
    /// [`defer_index_rebuild`](super::LoadOptions::defer_index_rebuild) choice:
    /// the same number when the indexes will be built, and that minus the
    /// **whole** index term when they will not — deferral does not build them,
    /// so nothing of that term is paid at load.
    ///
    /// This is what the ceiling compares, so that turning the deferral on
    /// actually buys headroom instead of being refused for memory it was never
    /// going to spend. It is also the honest answer to "would deferring get me
    /// under my ceiling?".
    pub fn projected_peak_bytes(&self, defer_index_rebuild: bool) -> u64 {
        if defer_index_rebuild {
            self.section_heap_bytes
                .saturating_add(self.transient_peak_bytes)
        } else {
            self.total_peak_bytes()
        }
    }
}

/// Per-cell cost for a column's declared type tag (the strings
/// `TypedColumn::type_tag` writes). An unrecognised tag — a column kind a newer
/// writer had and this build does not — is costed as a string, the most
/// expensive kind, rather than as zero.
fn cell_bytes(type_tag: &str) -> u64 {
    match type_tag {
        "bool" => BOOL_CELL_BYTES,
        "int64" | "float64" | "date" | "uniqueid" => NUMERIC_CELL_BYTES,
        _ => STRING_CELL_BYTES,
    }
}

/// Per-indexed-value cost for a column's declared type tag. Same
/// unknown-tag posture as [`cell_bytes`].
fn value_bytes(type_tag: &str) -> u64 {
    match type_tag {
        "int64" | "float64" | "date" | "uniqueid" | "bool" => SCALAR_VALUE_BYTES,
        _ => STRING_VALUE_BYTES,
    }
}

/// The column section for `node_type`, or `None` when the file declares no
/// section for it.
///
/// A node type with no stored properties has no column section, so an index
/// declared over it has no row count to scale by and contributes nothing. That
/// is an under-estimate rather than a wrong one: such a type's index is keyed
/// on `id`/`title`, which live in node slots, and the metadata carries no count
/// for them.
fn section_for<'a>(
    metadata: &'a FileMetadata,
    node_type: &str,
) -> Option<&'a PortableColumnSection> {
    metadata
        .column_sections
        .iter()
        .find(|section| section.type_name == node_type)
}

/// The declared type tag of `(node_type, property)`, or `None` when the file
/// records no column for it.
fn tag_for<'a>(metadata: &'a FileMetadata, node_type: &str, property: &str) -> Option<&'a str> {
    section_for(metadata, node_type)
        .and_then(|section| section.columns.get(property))
        .map(String::as_str)
}

/// Rows in `node_type`'s column section, or 0 when it has none.
fn rows_for(metadata: &FileMetadata, node_type: &str) -> u64 {
    section_for(metadata, node_type).map_or(0, |section| u64::from(section.row_count))
}

/// Per-row cost of one equality-shaped index (property, range, or composite)
/// over `properties`: the posting slot every row occupies, plus the key
/// material amortised over the assumed distinct-value density.
fn equality_index_row_bytes(
    metadata: &FileMetadata,
    node_type: &str,
    properties: &[String],
) -> u64 {
    let mut key_bytes: u64 = if properties.len() > 1 {
        COMPOSITE_KEY_VEC_BYTES
    } else {
        0
    };
    for property in properties {
        key_bytes += tag_for(metadata, node_type, property).map_or(STRING_VALUE_BYTES, value_bytes);
    }
    POSTING_ENTRY_BYTES + key_bytes / KEY_DENSITY_DIVISOR
}

/// Per-row cost of one unique constraint over `properties`: one map entry per
/// row, carrying the whole key rather than an amortised share of it.
fn unique_index_row_bytes(metadata: &FileMetadata, node_type: &str, properties: &[String]) -> u64 {
    let mut bytes = UNIQUE_ENTRY_BYTES + COMPOSITE_KEY_VEC_BYTES;
    for property in properties {
        bytes += tag_for(metadata, node_type, property).map_or(STRING_VALUE_BYTES, value_bytes);
    }
    bytes
}

/// The estimate for an already-parsed metadata block — the form the loader's
/// own ceiling check uses, so the number it refuses on is the number
/// [`estimate_load_memory`] reports for the same file.
pub(super) fn estimate_from_metadata(metadata: &FileMetadata) -> LoadMemoryEstimate {
    let mut node_rows: u64 = 0;
    let mut section_heap_bytes: u64 = 0;
    let mut compressed_total: u64 = metadata.topology_compressed_size;
    let mut largest_section: u64 = metadata.topology_compressed_size;

    for section in &metadata.column_sections {
        let rows = u64::from(section.row_count);
        node_rows += rows;
        section_heap_bytes += rows * NODE_ROW_BYTES;
        for type_tag in section.columns.values() {
            section_heap_bytes += rows * cell_bytes(type_tag);
        }
        compressed_total += section.compressed_size;
        largest_section = largest_section.max(section.compressed_size);
    }
    for optional in [
        metadata.embeddings_compressed_size,
        metadata.timeseries_compressed_size,
        metadata.secondary_labels_compressed_size,
        metadata.vector_index_compressed_size,
        metadata.text_index_compressed_size,
    ] {
        compressed_total += optional;
        largest_section = largest_section.max(optional);
    }

    // Floor: a decoded graph is never smaller than the bytes it decoded from.
    // It does not bind on any calibration fixture (the modelled term is 1.3×
    // to 9× above it); it exists so a corpus the constants above under-serve —
    // one column of very long strings, say — cannot be estimated at a fraction
    // of its own file size.
    section_heap_bytes = section_heap_bytes.max(compressed_total);

    let mut index_rebuild_bytes: u64 = 0;
    let mut declared_indexes: u32 = 0;
    for (node_type, property) in metadata
        .property_index_keys
        .iter()
        .chain(metadata.range_index_keys.iter())
    {
        declared_indexes += 1;
        index_rebuild_bytes += rows_for(metadata, node_type)
            * equality_index_row_bytes(metadata, node_type, std::slice::from_ref(property));
    }
    for (node_type, properties) in &metadata.composite_index_keys {
        declared_indexes += 1;
        index_rebuild_bytes += rows_for(metadata, node_type)
            * equality_index_row_bytes(metadata, node_type, properties);
    }
    for (node_type, properties) in &metadata.unique_constraint_keys {
        declared_indexes += 1;
        index_rebuild_bytes +=
            rows_for(metadata, node_type) * unique_index_row_bytes(metadata, node_type, properties);
    }

    LoadMemoryEstimate {
        index_rebuild_bytes,
        section_heap_bytes,
        transient_peak_bytes: largest_section * DECOMPRESS_RATIO_NUMERATOR
            / DECOMPRESS_RATIO_DENOMINATOR,
        node_rows,
        declared_indexes,
    }
}

/// Estimate what loading the `.kgl` at `path` will cost, reading only its
/// metadata head.
///
/// ```no_run
/// use kglite::api::io::estimate_load_memory;
///
/// let estimate = estimate_load_memory("graph.kgl")?;
/// println!(
///     "~{} MB settled, ~{} MB peak ({} MB of it index rebuild)",
///     estimate.total_settled_bytes() / 1_048_576,
///     estimate.total_peak_bytes() / 1_048_576,
///     estimate.index_rebuild_bytes / 1_048_576,
/// );
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// Portable `.kgl` containers only. A disk-graph *directory* has no single
/// metadata head to read and its indexes live on disk rather than being rebuilt
/// at load, so it is refused with `InvalidInput` rather than answered with a
/// number that would mean something else.
pub fn estimate_load_memory(path: &str) -> io::Result<LoadMemoryEstimate> {
    if std::path::Path::new(path).is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "'{path}' is a disk-mode graph directory, not a portable .kgl. Load memory is \
                 estimated from a .kgl's metadata head; a disk graph keeps its columns and \
                 indexes on disk and never rebuilds them at load, so the terms this reports \
                 would not describe it."
            ),
        ));
    }
    let head = super::read_metadata_head_from_file(path)?;
    Ok(estimate_from_metadata(&head))
}

/// [`estimate_load_memory`] against a `.kgl` byte buffer — the counterpart of
/// [`load_kgl_bytes`](super::load_kgl_bytes), and cheap for the same reason:
/// only the head of `data` is parsed.
pub fn estimate_load_memory_bytes(data: &[u8]) -> io::Result<LoadMemoryEstimate> {
    let head = super::read_metadata_head(data, "the byte buffer")?;
    Ok(estimate_from_metadata(&head))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// One measured fixture, reduced to what the estimator actually reads.
    ///
    /// The metadata is transcribed from the fixtures' own heads (dumped with
    /// `dev-docs/bench/scripts/make_indexed_fixture.py`'s output), and the
    /// footprint is the `cli / spaceeff` settled column of
    /// `dev-docs/bench/results/load-rss-2026-08-29.md` §1 — the tight-allocator
    /// route, which §0 shows is the one that measures the graph rather than the
    /// allocator.
    ///
    /// Recorded rather than regenerated: building a 500k-row graph inside a
    /// unit test would cost minutes, and the fixtures themselves live under
    /// `dev-docs/bench/out/`, which is gitignored and purged at 14 days. What
    /// this pins is the estimator's *arithmetic* against numbers that were
    /// measured once, which is what drifts.
    struct Fixture {
        rows: u32,
        columns: &'static [(&'static str, &'static str)],
        topology_compressed: u64,
        column_compressed: u64,
        measured_settled_bytes: u64,
    }

    const ITEM_COLUMNS: &[(&str, &str)] = &[
        ("category", "string"),
        ("count", "int64"),
        ("region", "string"),
        ("score", "float64"),
        ("sku", "string"),
    ];

    /// `indexed_500k.kgl`: 150.7 MB settled footprint, of which §4 attributes
    /// 64.1 MB to the four index structures.
    const INDEXED_500K: Fixture = Fixture {
        rows: 500_000,
        columns: ITEM_COLUMNS,
        topology_compressed: 2_659_663,
        column_compressed: 9_436_825,
        measured_settled_bytes: 150_700_000,
    };

    /// `noindex_500k.kgl`: byte-identical data, zero declared indexes,
    /// 86.6 MB settled footprint.
    const NOINDEX_500K: Fixture = Fixture {
        rows: 500_000,
        columns: ITEM_COLUMNS,
        topology_compressed: 2_659_663,
        column_compressed: 9_436_825,
        measured_settled_bytes: 86_600_000,
    };

    fn metadata_for(fixture: &Fixture, indexed: bool) -> FileMetadata {
        let mut metadata = FileMetadata {
            topology_compressed_size: fixture.topology_compressed,
            ..Default::default()
        };
        metadata.column_sections.push(PortableColumnSection {
            type_name: "Item".to_string(),
            compressed_size: fixture.column_compressed,
            row_count: fixture.rows,
            columns: fixture
                .columns
                .iter()
                .map(|(name, tag)| (name.to_string(), tag.to_string()))
                .collect::<HashMap<String, String>>(),
        });
        if indexed {
            metadata.property_index_keys = vec![
                ("Item".to_string(), "category".to_string()),
                ("Item".to_string(), "region".to_string()),
            ];
            metadata.composite_index_keys = vec![(
                "Item".to_string(),
                vec!["category".to_string(), "region".to_string()],
            )];
            metadata.unique_constraint_keys = vec![("Item".to_string(), vec!["sku".to_string()])];
        }
        metadata
    }

    /// The estimate must land within 0.5×–2× of what the same file was
    /// *measured* to cost. The band is wide because the section term is a
    /// heuristic whose own corpus spread is 0.56×–1.30× (module doc); it is
    /// narrow enough that a constant losing an order of magnitude, or a term
    /// silently going to zero, fails here.
    #[test]
    fn estimate_is_within_a_factor_of_two_of_measured_footprint() {
        for (name, fixture, indexed) in [
            ("indexed_500k", &INDEXED_500K, true),
            ("noindex_500k", &NOINDEX_500K, false),
        ] {
            let estimate = estimate_from_metadata(&metadata_for(fixture, indexed));
            let settled = estimate.total_settled_bytes() as f64;
            let measured = fixture.measured_settled_bytes as f64;
            let ratio = settled / measured;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "{name}: estimated {settled:.0} B against {measured:.0} B measured (×{ratio:.2})"
            );
        }
    }

    /// The index term is the modelled one, so it gets the tighter assertion:
    /// §4 measured it at 64.1 MB (tight allocator) / 79.1 MB (wheel), and the
    /// estimate must sit inside that band's neighbourhood rather than merely
    /// inside the factor-of-two gate above.
    #[test]
    fn index_term_matches_the_measured_index_term() {
        let indexed = estimate_from_metadata(&metadata_for(&INDEXED_500K, true));
        let noindex = estimate_from_metadata(&metadata_for(&NOINDEX_500K, false));

        assert_eq!(noindex.index_rebuild_bytes, 0, "no declaration, no term");
        assert_eq!(noindex.declared_indexes, 0);
        assert_eq!(indexed.declared_indexes, 4);
        // The two fixtures differ only in their declarations, so their section
        // terms must be identical — that is what makes the difference below
        // comparable to the measured difference between the two loads.
        assert_eq!(indexed.section_heap_bytes, noindex.section_heap_bytes);

        let term = indexed.index_rebuild_bytes as f64;
        assert!(
            (55e6..=95e6).contains(&term),
            "index term {term:.0} B is outside the 64.1-79.1 MB measured band's neighbourhood"
        );
    }

    /// Deferring the index rebuild removes exactly the modelled index term, so
    /// a caller can subtract it to decide whether the deferral gets them under
    /// a ceiling. Pinned because the refusal message advertises it.
    #[test]
    fn deferring_indexes_would_remove_the_index_term() {
        let indexed = estimate_from_metadata(&metadata_for(&INDEXED_500K, true));
        assert_eq!(
            indexed.total_settled_bytes() - indexed.index_rebuild_bytes,
            indexed.section_heap_bytes
        );
    }

    /// An index over a node type with no column section has no row count to
    /// scale by. It must contribute nothing rather than panic or count a
    /// phantom row — and it must still be counted as declared, so the refusal
    /// message's index count stays honest.
    #[test]
    fn an_index_on_a_type_with_no_section_contributes_nothing() {
        let mut metadata = metadata_for(&NOINDEX_500K, false);
        metadata.property_index_keys = vec![("Ghost".to_string(), "name".to_string())];
        let estimate = estimate_from_metadata(&metadata);
        assert_eq!(estimate.index_rebuild_bytes, 0);
        assert_eq!(estimate.declared_indexes, 1);
    }

    /// The transient is the *largest* section, not the sum: the load holds one
    /// decompression buffer at a time.
    #[test]
    fn transient_is_the_largest_single_section() {
        let estimate = estimate_from_metadata(&metadata_for(&NOINDEX_500K, false));
        let expected = NOINDEX_500K.column_compressed * DECOMPRESS_RATIO_NUMERATOR
            / DECOMPRESS_RATIO_DENOMINATOR;
        assert_eq!(estimate.transient_peak_bytes, expected);
        assert_eq!(
            estimate.total_peak_bytes(),
            estimate.total_settled_bytes() + expected
        );
    }

    /// An empty graph estimates to its own compressed size (the floor) rather
    /// than to zero, and reports no rows.
    #[test]
    fn an_empty_graph_falls_back_to_the_compressed_floor() {
        let metadata = FileMetadata {
            topology_compressed_size: 4096,
            ..Default::default()
        };
        let estimate = estimate_from_metadata(&metadata);
        assert_eq!(estimate.node_rows, 0);
        assert_eq!(estimate.section_heap_bytes, 4096);
    }

    /// A column kind this build does not know is costed as a string — the most
    /// expensive kind — rather than as free.
    #[test]
    fn an_unknown_column_tag_costs_a_string() {
        assert_eq!(cell_bytes("from-a-newer-writer"), STRING_CELL_BYTES);
        assert_eq!(value_bytes("from-a-newer-writer"), STRING_VALUE_BYTES);
    }
}
