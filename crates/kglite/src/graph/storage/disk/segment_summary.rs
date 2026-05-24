// src/graph/storage/disk/segment_summary.rs
//
// Per-segment RAM-resident summaries + segment manifest. PR1 phase 1
// of the disk-graph-improvement-plan (segmented CSR).
//
// Each segment of a segmented DiskGraph carries a ~1 KB summary
// loaded once at open and kept in RAM. The planner consults the
// manifest before iterating any segment so queries whose predicates
// correlate with segment boundaries (type filter, node-id range,
// conn-type filter) can prune the segments that cannot possibly
// contain a match.
//
// At Wikidata scale (~200 segments × ~1 KB) the whole manifest fits
// in ~200 KB — negligible against the per-PR2 heap → mmap savings.
//
// Not yet wired: this module defines types + serde only. Subsequent
// phases add:
//   - SegmentManifest emission during save (builder.rs, graph.rs)
//   - ArcSwap<Vec<Arc<Segment>>> hot-swap semantics on sealed-segment
//     addition
//   - Planner pruning in core/pattern_matching/matcher.rs
//   - Per-segment PropertyIndex path scheme
//
// Types intentionally use std-only primitives (HashSet<u64> instead
// of RoaringBitmap, simple NumericMinMax instead of BloomFilter) to
// keep the dependency surface flat. If subsequent phases show these
// are too heavy (expected only at Wikidata scale), we swap behind
// the same struct signatures.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;

/// Filename for the serialized segment manifest. Lives at the root
/// of a segmented .kgl directory alongside the per-segment subdirs.
pub const MANIFEST_FILE: &str = "seg_manifest.json";

/// Range-based predicate summary for one (NodeType, PropKey) tuple
/// inside a segment. Used by the planner to prune segments that
/// cannot contain a match for a given filter.
///
/// PR1 phase 1 ships only `NumericMinMax` — string predicates are
/// conservatively not pruned until the bloom-filter variant lands.
/// The enum shape is locked now so future additions are source-
/// compatible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropRange {
    /// Numeric column: closed min/max range.
    /// Pruning rule: segment may contain `x == v` iff `min <= v <= max`.
    NumericMinMax { min: f64, max: f64 },
    /// Placeholder for the future StringBloom variant — kept in the
    /// enum so match-exhaustiveness breaks obvious at the call site
    /// when the bloom path lands. Not emitted by phase 1.
    StringBloomPlaceholder,
}

// Manifest query methods (`PropRange::numeric` / `expand_with` /
// `might_contain_numeric`, plus `SegmentSummary::find_indexed_range` and the
// `SegmentManifest::candidates_for_*` helpers below) are kept under
// `#[allow(dead_code)]` because the manifest is built and persisted on every
// disk save but no executor path consults it yet — they form the API surface
// for a future segment-pruning optimiser pass.
#[allow(dead_code)]
impl PropRange {
    /// New numeric range covering a single observed value. Future calls
    /// to `expand_with` widen the min/max.
    pub fn numeric(value: f64) -> Self {
        PropRange::NumericMinMax {
            min: value,
            max: value,
        }
    }

    /// Widen this range to also cover `value`. No-op for non-numeric
    /// variants; a conservative choice while the bloom variant is
    /// stubbed out.
    pub fn expand_with(&mut self, value: f64) {
        if let PropRange::NumericMinMax { min, max } = self {
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
        }
    }

    /// Returns true if this range could contain `value` — i.e. the
    /// segment described by this summary might have a matching row.
    /// Conservative: unknown variants return `true` (cannot prune).
    pub fn might_contain_numeric(&self, value: f64) -> bool {
        match self {
            PropRange::NumericMinMax { min, max } => *min <= value && value <= *max,
            PropRange::StringBloomPlaceholder => true,
        }
    }
}

/// Per-segment summary. One instance per segment, held in RAM for
/// the lifetime of the DiskGraph.
///
/// Memory budget: ≤1 KB per segment with phase 1 types. ~200 KB
/// for a 200-segment Wikidata-scale graph — negligible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentSummary {
    /// Monotonically increasing segment identifier. Assigned at seal
    /// time by `SegmentManifest::append`.
    pub segment_id: u32,
    /// Inclusive-exclusive node-id range owned by this segment.
    /// `[lo, hi)`. Empty segments have `lo == hi`.
    pub node_id_lo: u32,
    pub node_id_hi: u32,
    /// Total edge count inside the segment. Sum across the manifest
    /// equals the graph's edge_count.
    pub edge_count: u64,
    /// Distinct `InternedKey` hashes of connection types appearing in
    /// this segment's edges. Planner uses this to prune for
    /// `MATCH ()-[r:TYPE]->()`.
    #[serde(default)]
    pub conn_types: HashSet<u64>,
    /// Per-node-type row counts in this segment. Keyed by NodeType
    /// `InternedKey` hash. Planner uses this for `MATCH (n:Type)`.
    #[serde(default)]
    pub node_type_counts: HashMap<u64, u32>,
    /// Indexed-property range summaries: `(node_type_hash, prop_hash,
    /// range)`. One tuple per indexed property that has any value in
    /// this segment. Planner consults for `MATCH (n:Type {prop: v})`.
    ///
    /// Stored as a Vec rather than a HashMap<(u64, u64), _> so the
    /// manifest can serde as JSON (serde_json rejects non-string map
    /// keys). Per-segment count is small (dozens, not thousands), so
    /// linear lookups via `find_indexed_range` are fine.
    #[serde(default)]
    pub indexed_prop_ranges: Vec<(u64, u64, PropRange)>,
}

impl SegmentSummary {
    /// Empty summary for a freshly-opened segment. Fields are filled
    /// during construction (walk edges, tally conn_types, compute
    /// min/max on indexed props) and sealed when save writes it to
    /// the manifest.
    pub fn new(segment_id: u32, node_id_lo: u32) -> Self {
        Self {
            segment_id,
            node_id_lo,
            node_id_hi: node_id_lo,
            edge_count: 0,
            conn_types: HashSet::new(),
            node_type_counts: HashMap::new(),
            indexed_prop_ranges: Vec::new(),
        }
    }

    /// Find the range summary for `(node_type, prop)` if present.
    /// Linear scan; per-segment list is small.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn find_indexed_range(&self, node_type_hash: u64, prop_hash: u64) -> Option<&PropRange> {
        self.indexed_prop_ranges
            .iter()
            .find(|(nt, p, _)| *nt == node_type_hash && *p == prop_hash)
            .map(|(_, _, r)| r)
    }

    /// True if this segment's [lo, hi) range covers the given node id.
    #[inline]
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn covers_node(&self, node_id: u32) -> bool {
        node_id >= self.node_id_lo && node_id < self.node_id_hi
    }

    /// True if this segment has at least one edge of the given type.
    /// Used by the planner to prune typed-edge matches.
    #[inline]
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn has_conn_type(&self, conn_type_hash: u64) -> bool {
        self.conn_types.contains(&conn_type_hash)
    }

    /// True if this segment has at least one node of the given type.
    #[inline]
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn has_node_type(&self, node_type_hash: u64) -> bool {
        self.node_type_counts.contains_key(&node_type_hash)
    }

    /// Might this segment contain a node of `node_type` with
    /// `prop == value`? Conservative: returns `true` when the index
    /// covers no data for this (type, prop) pair (meaning we can't
    /// rule the segment out).
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn might_match_numeric_prop(
        &self,
        node_type_hash: u64,
        prop_hash: u64,
        value: f64,
    ) -> bool {
        if !self.has_node_type(node_type_hash) {
            return false;
        }
        match self.find_indexed_range(node_type_hash, prop_hash) {
            Some(range) => range.might_contain_numeric(value),
            None => true,
        }
    }
}

/// List of all segment summaries for a DiskGraph. Loaded at open,
/// held in RAM, consulted by the planner before any scan.
///
/// Appending a segment is append-only: sealed segments are
/// immutable. Subsequent phases will wrap this in
/// `ArcSwap<Arc<SegmentManifest>>` so reader threads can hold a
/// stable snapshot while a writer publishes a new sealed segment.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    /// Segments in append order. The plan uses `segment_id` rather
    /// than position so future tooling can compact / reorder without
    /// changing IDs.
    pub segments: Vec<SegmentSummary>,
}

impl SegmentManifest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of segments.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// True when the manifest contains no segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Append a fresh, sealed segment. The caller owns
    /// `segment_id` assignment to stay compatible with whatever
    /// reordering / compaction we do later. Returns the index at
    /// which the segment was inserted.
    pub fn append(&mut self, summary: SegmentSummary) -> usize {
        let idx = self.segments.len();
        self.segments.push(summary);
        idx
    }

    /// Borrow a segment by position in the manifest. Prefer this over
    /// searching by `segment_id` on the hot path — the planner
    /// iterates in manifest order.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn get(&self, index: usize) -> Option<&SegmentSummary> {
        self.segments.get(index)
    }

    /// Iterate segments that *might* contain nodes in [lo, hi).
    /// Inclusive-exclusive. Returned in manifest order.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn candidates_for_node_range<'a>(
        &'a self,
        lo: u32,
        hi: u32,
    ) -> impl Iterator<Item = &'a SegmentSummary> + 'a {
        self.segments.iter().filter(move |s| {
            // Overlap test: [s.node_id_lo, s.node_id_hi) ∩ [lo, hi)
            s.node_id_hi > lo && s.node_id_lo < hi
        })
    }

    /// Iterate segments that *might* contain edges of `conn_type`.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn candidates_for_conn_type<'a>(
        &'a self,
        conn_type_hash: u64,
    ) -> impl Iterator<Item = &'a SegmentSummary> + 'a {
        self.segments
            .iter()
            .filter(move |s| s.has_conn_type(conn_type_hash))
    }

    /// Iterate segments that *might* contain nodes of `node_type`.
    #[allow(dead_code)] // Future segment-pruning optimiser; see PropRange comment above.
    pub fn candidates_for_node_type<'a>(
        &'a self,
        node_type_hash: u64,
    ) -> impl Iterator<Item = &'a SegmentSummary> + 'a {
        self.segments
            .iter()
            .filter(move |s| s.has_node_type(node_type_hash))
    }

    /// Persist the manifest as pretty-printed JSON at `dir/seg_manifest.json`.
    pub fn save_to(&self, dir: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        std::fs::write(dir.join(MANIFEST_FILE), json)
    }

    /// Load a manifest from `dir/seg_manifest.json`. Returns an empty
    /// manifest when the file is absent — legacy single-file graphs
    /// are treated as a one-segment graph by the eventual reader.
    pub fn load_from(dir: &Path) -> io::Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(Self::new());
        }
        let json = std::fs::read_to_string(&path)?;
        serde_json::from_str(&json).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn prop_range_numeric_expands_and_contains() {
        let mut r = PropRange::numeric(5.0);
        r.expand_with(10.0);
        r.expand_with(2.5);
        assert!(matches!(
            r,
            PropRange::NumericMinMax {
                min: 2.5,
                max: 10.0
            }
        ));
        assert!(r.might_contain_numeric(5.0));
        assert!(r.might_contain_numeric(2.5));
        assert!(r.might_contain_numeric(10.0));
        assert!(!r.might_contain_numeric(11.0));
        assert!(!r.might_contain_numeric(2.0));
    }

    #[test]
    fn prop_range_placeholder_never_prunes() {
        assert!(PropRange::StringBloomPlaceholder.might_contain_numeric(0.0));
        assert!(PropRange::StringBloomPlaceholder.might_contain_numeric(1e18));
    }

    #[test]
    fn segment_summary_tracks_range() {
        let s = SegmentSummary {
            segment_id: 0,
            node_id_lo: 100,
            node_id_hi: 200,
            ..SegmentSummary::new(0, 100)
        };
        assert!(!s.covers_node(99));
        assert!(s.covers_node(100));
        assert!(s.covers_node(199));
        assert!(!s.covers_node(200));
    }

    #[test]
    fn manifest_append_and_filter() {
        let mut m = SegmentManifest::new();
        let mut s0 = SegmentSummary::new(0, 0);
        s0.node_id_hi = 100;
        s0.conn_types.insert(42);
        s0.node_type_counts.insert(7, 50);
        let mut s1 = SegmentSummary::new(1, 100);
        s1.node_id_hi = 200;
        s1.conn_types.insert(99);
        s1.node_type_counts.insert(7, 75);
        m.append(s0);
        m.append(s1);
        assert_eq!(m.len(), 2);

        // Node range pruning
        let hits: Vec<_> = m.candidates_for_node_range(50, 120).collect();
        assert_eq!(hits.len(), 2); // overlaps both
        let hits: Vec<_> = m.candidates_for_node_range(150, 180).collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].segment_id, 1);

        // Conn-type pruning
        let hits: Vec<_> = m.candidates_for_conn_type(42).collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].segment_id, 0);

        // Node-type pruning — both segments have type 7
        let hits: Vec<_> = m.candidates_for_node_type(7).collect();
        assert_eq!(hits.len(), 2);
        let hits: Vec<_> = m.candidates_for_node_type(1234).collect();
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn indexed_prop_prunes_out_of_range() {
        let mut s = SegmentSummary::new(0, 0);
        s.node_id_hi = 100;
        s.node_type_counts.insert(7, 10);
        s.indexed_prop_ranges.push((
            7,
            99,
            PropRange::NumericMinMax {
                min: 10.0,
                max: 20.0,
            },
        ));
        assert!(s.might_match_numeric_prop(7, 99, 15.0));
        assert!(!s.might_match_numeric_prop(7, 99, 25.0));
        // Different node type — pruned out.
        assert!(!s.might_match_numeric_prop(123, 99, 15.0));
        // Indexed prop not present in this segment — conservative accept.
        assert!(s.might_match_numeric_prop(7, 77, 15.0));
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = TempDir::new().unwrap();
        let mut m = SegmentManifest::new();
        let mut s = SegmentSummary::new(0, 0);
        s.node_id_hi = 1000;
        s.edge_count = 5000;
        s.conn_types.insert(42);
        s.node_type_counts.insert(7, 500);
        s.indexed_prop_ranges.push((
            7,
            99,
            PropRange::NumericMinMax {
                min: 1.0,
                max: 99.0,
            },
        ));
        m.append(s);

        m.save_to(tmp.path()).unwrap();
        let loaded = SegmentManifest::load_from(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 1);
        let s2 = &loaded.segments[0];
        assert_eq!(s2.node_id_hi, 1000);
        assert_eq!(s2.edge_count, 5000);
        assert!(s2.has_conn_type(42));
        assert_eq!(s2.node_type_counts.get(&7), Some(&500));
        assert!(s2.find_indexed_range(7, 99).is_some());
    }

    #[test]
    fn missing_file_loads_as_empty() {
        let tmp = TempDir::new().unwrap();
        let m = SegmentManifest::load_from(tmp.path()).unwrap();
        assert!(m.is_empty());
    }
}
