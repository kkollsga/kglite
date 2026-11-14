//! `FrozenGraph`: a read-only topology served straight from the raw sections of
//! a v9 `.kgl`, for a graph loaded into mapped storage.
//!
//! A heap graph holds every node and edge as a petgraph entry (about 40 B per
//! node and 32 B per edge). A frozen graph holds only what the file maps: the
//! `NodeData` records (see [`node_record`]), an out- and an in-adjacency in CSR
//! form, and one endpoint record per edge slot. Resident memory is therefore
//! the working set of the pages a query touches, not the size of the graph.
//!
//! ## Identity
//!
//! Node and edge indices are the slots of the graph that was saved, tombstones
//! included: slot `i` of the arrays is `NodeIndex::new(i)` / `EdgeIndex::new(i)`.
//! A vacant node is a clear bit in the alive bitset (absent when no slot is
//! vacant); a vacant edge has `src == VACANT` in its endpoint record. The free
//! lists are not stored, because a heap load does not preserve them either:
//! petgraph's deserializer rebuilds both from the vacancy pattern alone, highest
//! slot first, and [`FrozenGraph::to_mapped`] and [`FrozenGraph::slot_mirror`]
//! reproduce exactly that order.
//!
//! ## Adjacency
//!
//! `out_off`/`in_off` hold `node_bound + 1` offsets into the matching
//! `*_edges` (peer node and edge slot) and `*_ty` (connection-type ordinal)
//! arrays. Each node's run is sorted by type ordinal, then by edge slot
//! descending (petgraph's order for a single-type graph: newest edge first), so
//! a typed expansion binary-searches the run and touches only its own edges.
//! Ordinals index [`FrozenGraph::types`], which is ascending by key.
//!
//! ## Writes
//!
//! A frozen graph is never mutated. The first write to a graph holding one
//! puts a copy-on-write overlay over it (`GraphBackend::overlay_frozen`,
//! `ForkBase::Frozen`): the overlay holds what was written and reads the rest
//! from here, so the heap grows with the writes, not the graph. It is the same
//! overlay a held reader gets over a heap graph, except that it never folds
//! back. [`FrozenGraph::to_mapped`] is the way out, taken once by the promote
//! valve or a whole-graph operation; a clone shares the frozen data by `Arc`.

pub(crate) mod array;
mod iters;
pub(crate) mod node_record;
mod promote;
mod read;

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, OnceLock, RwLock};

use petgraph::graph::NodeIndex;
use petgraph::Direction;
use rustc_hash::FxHashMap;

pub(crate) use iters::{
    FrozenEdgeIndices, FrozenEdgeRefs, FrozenEdges, FrozenEdgesConnecting, FrozenNeighbors,
    FrozenNodeIndices,
};

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::disk::csr::CsrEdge;
use crate::graph::storage::mapped::mmap_vec::MmapPod;

/// Longest sorted array (a node's type run, the type table) searched by forward scan rather than
/// binary search.
const SCAN_RUN_MAX: usize = 16;
use crate::graph::storage::mapped::peer_base::PeerBase;
use crate::graph::storage::mapped::MappedTypeIndex;
use crate::graph::storage::slot_mirror::SlotMirror;
use array::{FrozenArray, FrozenElement};
use node_record::NodeSlab;

/// `src` of an edge slot that holds no edge.
pub(crate) const VACANT: u32 = u32::MAX;

/// One edge slot: its endpoints and connection-type ordinal.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct FrozenEndpoint {
    pub(crate) src: u32,
    pub(crate) dst: u32,
    pub(crate) ty: u16,
    pub(crate) pad: u16,
}

const _: () = assert!(std::mem::size_of::<FrozenEndpoint>() == 12);
const _: () = assert!(std::mem::align_of::<FrozenEndpoint>() == 4);

// SAFETY: repr(C), 12 bytes with no implicit padding (`pad` is explicit), and
// every field accepts every bit pattern.
unsafe impl MmapPod for FrozenEndpoint {}

impl FrozenElement for FrozenEndpoint {
    const WIDTH: usize = 12;
    fn decode(bytes: &[u8]) -> Self {
        FrozenEndpoint {
            src: u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")),
            dst: u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")),
            ty: u16::from_le_bytes(bytes[8..10].try_into().expect("2 bytes")),
            pad: u16::from_le_bytes(bytes[10..12].try_into().expect("2 bytes")),
        }
    }
}

impl FrozenElement for CsrEdge {
    const WIDTH: usize = 8;
    fn decode(bytes: &[u8]) -> Self {
        CsrEdge {
            peer: u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")),
            edge_idx: u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")),
        }
    }
}

/// The arrays a frozen graph is built from, before validation.
pub(crate) struct FrozenParts {
    pub(crate) nodes: NodeSlab,
    pub(crate) node_count: usize,
    /// `None` when every slot below `nodes.len()` is occupied.
    pub(crate) node_alive: Option<FrozenArray<u64>>,
    pub(crate) out_off: FrozenArray<u32>,
    pub(crate) in_off: FrozenArray<u32>,
    pub(crate) out_edges: FrozenArray<CsrEdge>,
    pub(crate) in_edges: FrozenArray<CsrEdge>,
    pub(crate) out_ty: FrozenArray<u16>,
    pub(crate) in_ty: FrozenArray<u16>,
    pub(crate) endpoints: FrozenArray<FrozenEndpoint>,
    /// Per-node type presence masks, `[out, in]` per node; see `FrozenGraph::type_mask`.
    pub(crate) type_mask: Option<FrozenArray<u32>>,
    pub(crate) edge_count: usize,
    /// Connection types, strictly ascending by key; ordinals index this.
    pub(crate) types: Vec<InternedKey>,
    /// Edges that carry properties, by edge slot.
    pub(crate) edge_props: FxHashMap<u32, EdgeData>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid frozen topology: {}", message.into()),
    )
}

/// Read-only topology over the raw sections of a `.kgl`. See the module docs.
pub struct FrozenGraph {
    nodes: NodeSlab,
    node_count: usize,
    node_alive: Option<FrozenArray<u64>>,
    out_off: FrozenArray<u32>,
    in_off: FrozenArray<u32>,
    out_edges: FrozenArray<CsrEdge>,
    in_edges: FrozenArray<CsrEdge>,
    out_ty: FrozenArray<u16>,
    in_ty: FrozenArray<u16>,
    endpoints: FrozenArray<FrozenEndpoint>,
    /// `[out, in]` presence masks per node: bit `ordinal % 32` is set when the
    /// node has an edge of that type in that direction. A clear bit proves the
    /// typed run empty with one load; a set bit may be a collision (more than 32
    /// types) and falls through to the run lookup. `None` for a file without them.
    type_mask: Option<FrozenArray<u32>>,
    edge_count: usize,
    types: Vec<InternedKey>,
    /// One property-less `EdgeData` per ordinal, so an edge reference can
    /// borrow its weight without a heap record per edge.
    templates: Vec<EdgeData>,
    edge_props: Arc<FxHashMap<u32, EdgeData>>,
    /// `!edge_props.is_empty()`, read directly so an edge without properties
    /// costs one byte load, not a walk through the map's `Arc`.
    has_edge_props: bool,
    pub(crate) column_stores: FxHashMap<InternedKey, Arc<ColumnStore>>,
    /// Peer counts the file recorded; see `mapped::peer_base`.
    peer_base: Option<Arc<PeerBase>>,
    /// Per-connection-type peer-count blocks, filled on first use.
    type_index: RwLock<HashMap<u64, Arc<MappedTypeIndex>>>,
    /// The free lists a heap load of this graph would have, built on the first
    /// write (an overlay allocates by this prediction); see [`Self::slot_mirror`].
    slot_mirror: OnceLock<SlotMirror>,
    /// The key a record with an out-of-range type ordinal reads as; see
    /// [`Self::type_key`]. Unset (the default key) for a graph built outside a load.
    invalid_type: InternedKey,
}

impl FrozenGraph {
    /// Check `parts` against each other and build the graph.
    ///
    /// Every check is O(1) or one sequential pass over the offset and bitset
    /// arrays (never over the edge arrays); the node records were validated
    /// when `parts.nodes` was built. The endpoint array is not walked here, so
    /// a CRC-valid file whose adjacency arrays or endpoint records are
    /// inconsistent can yield wrong answers or absent rows. The readers still
    /// never panic (an unknown type ordinal reads as [`Self::invalid_type`],
    /// an unknown template as an empty one) and never index out of bounds
    /// (array reads use `get`).
    pub(crate) fn new(parts: FrozenParts) -> io::Result<Self> {
        let FrozenParts {
            nodes,
            node_count,
            node_alive,
            out_off,
            in_off,
            out_edges,
            in_edges,
            out_ty,
            in_ty,
            endpoints,
            type_mask,
            edge_count,
            types,
            edge_props,
        } = parts;
        let bound = nodes.as_slice().len();
        if bound >= u32::MAX as usize || endpoints.len() >= u32::MAX as usize {
            return Err(invalid("more slots than a 32-bit index can name"));
        }
        if out_off.len() != bound + 1 || in_off.len() != bound + 1 {
            return Err(invalid(
                "an offset array does not have one entry per node + 1",
            ));
        }
        for (edges, tys) in [(&out_edges, &out_ty), (&in_edges, &in_ty)] {
            if edges.len() != edge_count || tys.len() != edge_count {
                return Err(invalid(
                    "an adjacency array does not hold one entry per edge",
                ));
            }
        }
        if type_mask.as_ref().is_some_and(|m| m.len() != 2 * bound) {
            return Err(invalid("the type mask does not hold two entries per node"));
        }
        if edge_count > endpoints.len() {
            return Err(invalid("more edges than edge slots"));
        }
        if edge_count > 0 && (bound == 0 || types.is_empty()) {
            return Err(invalid("edges without nodes or connection types"));
        }
        if types.len() > u16::MAX as usize || types.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("connection types are not strictly ascending"));
        }
        for off in [&out_off, &in_off] {
            let slice = off.as_slice();
            if slice.first() != Some(&0)
                || slice.last().copied() != Some(edge_count as u32)
                || slice.windows(2).any(|w| w[0] > w[1])
            {
                return Err(invalid(
                    "an offset array is not a monotone run over the edges",
                ));
            }
        }
        match &node_alive {
            Some(words) => {
                if words.len() != bound.div_ceil(64) {
                    return Err(invalid("the alive bitset does not cover the node array"));
                }
                let ones: usize = words
                    .as_slice()
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum();
                let tail_bits = bound % 64;
                let tail_clear =
                    tail_bits == 0 || words.as_slice().last().is_none_or(|w| w >> tail_bits == 0);
                if ones != node_count || !tail_clear {
                    return Err(invalid("the alive bitset disagrees with the node count"));
                }
            }
            None => {
                if node_count != bound {
                    return Err(invalid("vacant nodes without an alive bitset"));
                }
            }
        }
        let templates = types
            .iter()
            .map(|key| EdgeData::new_interned(*key, Vec::new()))
            .collect();
        Ok(FrozenGraph {
            nodes,
            node_count,
            node_alive,
            out_off,
            in_off,
            out_edges,
            in_edges,
            out_ty,
            in_ty,
            endpoints,
            type_mask,
            edge_count,
            types,
            templates,
            has_edge_props: !edge_props.is_empty(),
            edge_props: Arc::new(edge_props),
            column_stores: FxHashMap::default(),
            peer_base: None,
            type_index: RwLock::new(HashMap::new()),
            slot_mirror: OnceLock::new(),
            invalid_type: InternedKey::default(),
        })
    }

    /// Name the key an out-of-range type ordinal reads as. A load interns it,
    /// so a consumer that resolves the key of any edge gets a name, not a panic.
    pub(crate) fn set_invalid_type(&mut self, key: InternedKey) {
        self.invalid_type = key;
    }

    /// The free lists petgraph's deserializer builds from this graph's vacancy
    /// pattern, which is exactly what [`Self::to_stable_graph`] produces, so an
    /// overlay allocating by this prediction lands on the slots its fold into
    /// that graph hands out. Built on first use: a read-only session never pays
    /// for it, and a graph with no vacancy never scans.
    ///
    /// The scan is skipped when the counts show no hole and every node is
    /// alive. A crafted file whose endpoints name a slot past the node bound
    /// would then disagree with `to_stable_graph` (which treats such an edge as
    /// vacant), and the fold that follows refuses it: an abort of that write,
    /// never an out-of-bounds read.
    pub(crate) fn slot_mirror(&self) -> &SlotMirror {
        self.slot_mirror.get_or_init(|| {
            let no_holes = self.node_alive.is_none() && self.edge_count == self.endpoints.len();
            if no_holes {
                return SlotMirror::from_vacancies(Vec::new(), Vec::new());
            }
            let bound = self.node_bound_len();
            let free_nodes = (0..bound as u32)
                .filter(|&slot| !self.node_is_alive(slot as usize))
                .collect();
            let free_edges = (0..self.endpoints.len() as u32)
                .filter(|&slot| !self.edge_is_live(slot as usize))
                .collect();
            SlotMirror::from_vacancies(free_nodes, free_edges)
        })
    }

    /// Whether edge `slot` holds an edge in the heap graph this one converts
    /// to: an endpoint record whose both ends are live nodes.
    pub(crate) fn edge_is_live(&self, slot: usize) -> bool {
        let Some(ep) = self.endpoint(slot) else {
            return false;
        };
        let bound = self.node_bound_len();
        (ep.src as usize) < bound
            && (ep.dst as usize) < bound
            && self.node_is_alive(ep.src as usize)
            && self.node_is_alive(ep.dst as usize)
    }

    #[inline(always)]
    pub(crate) fn node_bound_len(&self) -> usize {
        self.nodes.as_slice().len()
    }

    #[inline(always)]
    pub(crate) fn node_slice(&self) -> &[NodeData] {
        self.nodes.as_slice()
    }

    #[inline(always)]
    pub(crate) fn has_vacant_nodes(&self) -> bool {
        self.node_alive.is_some()
    }

    #[inline(always)]
    pub(crate) fn node_is_alive(&self, index: usize) -> bool {
        match &self.node_alive {
            None => index < self.nodes.as_slice().len(),
            Some(words) => words
                .as_slice()
                .get(index >> 6)
                .is_some_and(|word| word >> (index & 63) & 1 == 1),
        }
    }

    /// The endpoint record of edge slot `index`, or `None` for a vacant or
    /// absent slot.
    #[inline(always)]
    pub(crate) fn endpoint(&self, index: usize) -> Option<FrozenEndpoint> {
        self.endpoints
            .as_slice()
            .get(index)
            .copied()
            .filter(|ep| ep.src != VACANT)
    }

    #[inline(always)]
    pub(crate) fn endpoint_slots(&self) -> &[FrozenEndpoint] {
        self.endpoints.as_slice()
    }

    /// The weight of edge slot `index`, whose type ordinal is `ty`.
    #[inline(always)]
    pub(crate) fn edge_data(&self, index: u32, ty: u16) -> &EdgeData {
        if self.has_edge_props {
            if let Some(weight) = self.edge_props.get(&index) {
                return weight;
            }
        }
        self.templates
            .get(usize::from(ty))
            .unwrap_or_else(|| self.fallback_template())
    }

    /// The template an out-of-range ordinal reads as (a crafted file): the
    /// first one, or an empty edge when the file lists no types.
    #[cold]
    fn fallback_template(&self) -> &EdgeData {
        static EMPTY: OnceLock<EdgeData> = OnceLock::new();
        self.templates.first().unwrap_or_else(|| {
            EMPTY.get_or_init(|| EdgeData::new_interned(InternedKey::default(), Vec::new()))
        })
    }

    /// The connection type of ordinal `ty`. An ordinal the file does not list
    /// (only a crafted one) reads as [`Self::invalid_type`], never panics.
    #[inline(always)]
    pub(crate) fn type_key(&self, ty: u16) -> InternedKey {
        self.types
            .get(usize::from(ty))
            .copied()
            .unwrap_or(self.invalid_type)
    }

    /// The ordinal of `key`, if any edge of the file has that type.
    #[inline]
    pub(crate) fn ordinal_of(&self, key: InternedKey) -> Option<u16> {
        // A traversal asks for the same key at every node, so the scan's exit branch is
        // predicted and costs less than a binary search's dependent load chain.
        if self.types.len() <= SCAN_RUN_MAX {
            return self.types.iter().position(|&k| k == key).map(|i| i as u16);
        }
        self.types.binary_search(&key).ok().map(|i| i as u16)
    }

    /// Every live edge as `(source, target, connection type)`.
    #[inline(never)]
    pub(crate) fn for_each_endpoint_key<F: FnMut(NodeIndex, NodeIndex, InternedKey)>(
        &self,
        mut f: F,
    ) {
        for ep in self.endpoint_slots().iter().filter(|ep| ep.src != VACANT) {
            f(
                NodeIndex::new(ep.src as usize),
                NodeIndex::new(ep.dst as usize),
                self.type_key(ep.ty),
            );
        }
    }

    /// Every live edge of `conn_type` as `(source, target, slot, properties)`,
    /// until `f` returns `false`.
    #[inline(never)]
    pub(crate) fn for_each_edge_of_type<F>(&self, conn_type: InternedKey, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32, &[(InternedKey, crate::datatypes::Value)]) -> bool,
    {
        let Some(ordinal) = self.ordinal_of(conn_type) else {
            return;
        };
        for (slot, ep) in self.endpoint_slots().iter().enumerate() {
            if ep.src != VACANT
                && ep.ty == ordinal
                && !f(
                    NodeIndex::new(ep.src as usize),
                    NodeIndex::new(ep.dst as usize),
                    slot as u32,
                    self.edge_data(slot as u32, ep.ty).properties.as_slice(),
                )
            {
                return;
            }
        }
    }

    /// The CSR run of `node` in `dir`, as positions in the matching arrays.
    #[inline(always)]
    pub(crate) fn run(&self, node: usize, dir: Direction) -> (usize, usize) {
        let off = match dir {
            Direction::Outgoing => self.out_off.as_slice(),
            Direction::Incoming => self.in_off.as_slice(),
        };
        match (off.get(node), off.get(node + 1)) {
            (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
            _ => (0, 0),
        }
    }

    #[inline(always)]
    pub(crate) fn csr(&self, dir: Direction) -> (&[CsrEdge], &[u16]) {
        match dir {
            Direction::Outgoing => (self.out_edges.as_slice(), self.out_ty.as_slice()),
            Direction::Incoming => (self.in_edges.as_slice(), self.in_ty.as_slice()),
        }
    }

    /// The offsets, adjacency and type ordinals of one direction, resolved by a single branch.
    #[inline(always)]
    pub(crate) fn dir_arrays(&self, dir: Direction) -> (&[u32], &[CsrEdge], &[u16]) {
        match dir {
            Direction::Outgoing => (
                self.out_off.as_slice(),
                self.out_edges.as_slice(),
                self.out_ty.as_slice(),
            ),
            Direction::Incoming => (
                self.in_off.as_slice(),
                self.in_edges.as_slice(),
                self.in_ty.as_slice(),
            ),
        }
    }

    /// Whether `node` may hold an edge of type `ordinal` in `dir`: `false` is a
    /// proof of absence, `true` only means the run must be searched.
    #[inline(always)]
    pub(crate) fn may_have_type(&self, node: usize, dir: Direction, ordinal: u16) -> bool {
        let Some(mask) = &self.type_mask else {
            return true;
        };
        let at = 2 * node + usize::from(dir == Direction::Incoming);
        match mask.as_slice().get(at) {
            Some(bits) => bits >> (ordinal & 31) & 1 == 1,
            None => true,
        }
    }

    /// `run`, narrowed to the entries of type ordinal `ordinal` by binary
    /// search over the type array (the run is sorted by ordinal).
    pub(crate) fn typed_run(&self, node: usize, dir: Direction, ordinal: u16) -> (usize, usize) {
        let (off, _, tys) = self.dir_arrays(dir);
        Self::narrow(off, tys, node, ordinal)
    }

    #[inline(always)]
    fn narrow(off: &[u32], tys: &[u16], node: usize, ordinal: u16) -> (usize, usize) {
        let (lo, hi) = match (off.get(node), off.get(node + 1)) {
            (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
            _ => (0, 0),
        };
        let Some(run) = tys.get(lo..hi) else {
            return (0, 0);
        };
        // A typical run is a handful of entries, where a forward scan beats two binary searches.
        let (start, end) = if run.len() <= SCAN_RUN_MAX {
            let start = run.iter().position(|&t| t >= ordinal).unwrap_or(run.len());
            let len = run[start..].iter().take_while(|&&t| t == ordinal).count();
            (start, start + len)
        } else {
            let start = run.partition_point(|&t| t < ordinal);
            (
                start,
                start + run[start..].partition_point(|&t| t <= ordinal),
            )
        };
        (lo + start, lo + end)
    }

    pub(crate) fn live_node_count(&self) -> usize {
        self.node_count
    }

    pub(crate) fn live_edge_count(&self) -> usize {
        self.edge_count
    }

    pub(crate) fn edge_bound_len(&self) -> usize {
        self.endpoints.len()
    }

    /// Install a peer-count base the file recorded; the counts describe this
    /// topology exactly, and a frozen graph has no edge writes to outdate them.
    pub(crate) fn install_peer_base(&mut self, base: PeerBase) {
        if let Ok(mut map) = self.type_index.write() {
            map.clear();
        }
        self.peer_base = (!base.is_empty()).then(|| Arc::new(base));
    }

    pub(crate) fn peer_base(&self) -> Option<&Arc<PeerBase>> {
        self.peer_base.as_ref()
    }

    pub(crate) fn install_column_store(&mut self, key: InternedKey, store: Arc<ColumnStore>) {
        self.column_stores.insert(key, store);
    }

    pub(crate) fn column_store_mut(&mut self, key: InternedKey) -> Option<&mut Arc<ColumnStore>> {
        self.column_stores.get_mut(&key)
    }

    pub(crate) fn take_column_store(&mut self, key: InternedKey) -> Option<Arc<ColumnStore>> {
        self.column_stores.remove(&key)
    }

    pub(crate) fn clear_column_stores(&mut self) {
        self.column_stores.clear();
    }

    /// Fetch or create the per-connection-type block of peer counts.
    pub(crate) fn ensure_type_index(&self, conn_type: InternedKey) -> Arc<MappedTypeIndex> {
        let key = conn_type.as_u64();
        if let Ok(map) = self.type_index.read() {
            if let Some(block) = map.get(&key) {
                return Arc::clone(block);
            }
        }
        let built = Arc::new(MappedTypeIndex {
            persisted: self
                .peer_base
                .as_ref()
                .and_then(|base| base.get(key))
                .cloned(),
            ..Default::default()
        });
        let Ok(mut map) = self.type_index.write() else {
            return built;
        };
        Arc::clone(map.entry(key).or_insert_with(|| Arc::clone(&built)))
    }

    /// Edge counts of `conn_type` grouped by the far end of `dir`
    /// (`Outgoing` groups by target, `Incoming` by source), shared until the
    /// graph is dropped or promoted.
    pub(crate) fn peer_counts(
        &self,
        conn_type: InternedKey,
        dir: Direction,
    ) -> Arc<HashMap<u32, i64>> {
        let block = self.ensure_type_index(conn_type);
        let wanted = match dir {
            Direction::Outgoing => &block.out_peer_counts,
            Direction::Incoming => &block.in_peer_counts,
        };
        Arc::clone(wanted.get_or_init(|| match &block.persisted {
            Some(persisted) => Arc::new(persisted.counts(dir)),
            None => Arc::new(self.scan_peer_counts(conn_type, dir)),
        }))
    }

    /// Group the edges of `conn_type` by the far end of `dir` from the CSR
    /// runs: an outgoing aggregate groups by target, so it counts each node's
    /// incoming run, and vice versa.
    fn scan_peer_counts(&self, conn_type: InternedKey, dir: Direction) -> HashMap<u32, i64> {
        let mut counts = HashMap::new();
        let Some(ordinal) = self.ordinal_of(conn_type) else {
            return counts;
        };
        let scanned = match dir {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
        };
        for node in 0..self.node_bound_len() {
            let (lo, hi) = self.typed_run(node, scanned, ordinal);
            if hi > lo {
                counts.insert(node as u32, (hi - lo) as i64);
            }
        }
        counts
    }

    /// The distinct sources of `conn_type`, ascending, at most `max`.
    pub(crate) fn sources_of(&self, conn_type: InternedKey, max: Option<usize>) -> Vec<u32> {
        if let Some(base) = &self.peer_base {
            if let Some(peers) = base.get(conn_type.as_u64()) {
                return peers.sources(max);
            }
        }
        let Some(ordinal) = self.ordinal_of(conn_type) else {
            return Vec::new();
        };
        let limit = max.unwrap_or(usize::MAX);
        let mut sources = Vec::new();
        for node in 0..self.node_bound_len() {
            if sources.len() >= limit {
                break;
            }
            let (lo, hi) = self.typed_run(node, Direction::Outgoing, ordinal);
            if hi > lo {
                sources.push(node as u32);
            }
        }
        sources
    }

    /// Check the CRC32 of every section this graph serves in place.
    pub(crate) fn verify_sections(&self) -> io::Result<()> {
        for section in self.shared_sections() {
            section.verify()?;
        }
        Ok(())
    }

    /// Drop the resident pages of every mapped section, after a load-time pass
    /// (checksum, record validation, the row check) touched them. The pages are
    /// clean and refault from the file on first use.
    pub(crate) fn release_resident(&self) {
        for section in self.shared_sections() {
            section.release_resident();
        }
    }

    /// Bytes of this topology read in place from mapped `.kgl` sections.
    pub(crate) fn mapped_bytes(&self) -> usize {
        self.shared_sections()
            .iter()
            .map(|section| section.bytes().len())
            .sum()
    }

    /// The mapped `.kgl` sections behind this topology.
    pub(crate) fn shared_sections(
        &self,
    ) -> Vec<&Arc<crate::graph::io::file::mappable::SharedSection>> {
        [
            self.nodes.shared_section(),
            self.node_alive.as_ref().and_then(|a| a.shared_section()),
            self.out_off.shared_section(),
            self.in_off.shared_section(),
            self.out_edges.shared_section(),
            self.in_edges.shared_section(),
            self.out_ty.shared_section(),
            self.in_ty.shared_section(),
            self.endpoints.shared_section(),
            self.type_mask.as_ref().and_then(|a| a.shared_section()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Bytes of this topology held on the heap: the property-less templates,
    /// the sparse edge-property map, and any array a load had to copy.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.nodes.heap_bytes()
            + self.node_alive.as_ref().map_or(0, |a| a.heap_bytes())
            + self.out_off.heap_bytes()
            + self.in_off.heap_bytes()
            + self.out_edges.heap_bytes()
            + self.in_edges.heap_bytes()
            + self.out_ty.heap_bytes()
            + self.in_ty.heap_bytes()
            + self.endpoints.heap_bytes()
            + self.type_mask.as_ref().map_or(0, |m| m.heap_bytes())
            + self.templates.len() * std::mem::size_of::<EdgeData>()
            + self.edge_props.len() * 64
    }
}

impl Clone for FrozenGraph {
    /// Shares every mapped array and copies the small heap state; the
    /// peer-count blocks are caches and start empty.
    fn clone(&self) -> Self {
        FrozenGraph {
            nodes: self.nodes.clone(),
            node_count: self.node_count,
            node_alive: self.node_alive.clone(),
            out_off: self.out_off.clone(),
            in_off: self.in_off.clone(),
            out_edges: self.out_edges.clone(),
            in_edges: self.in_edges.clone(),
            out_ty: self.out_ty.clone(),
            in_ty: self.in_ty.clone(),
            endpoints: self.endpoints.clone(),
            type_mask: self.type_mask.clone(),
            edge_count: self.edge_count,
            types: self.types.clone(),
            templates: self.templates.clone(),
            edge_props: Arc::clone(&self.edge_props),
            has_edge_props: self.has_edge_props,
            column_stores: self.column_stores.clone(),
            peer_base: self.peer_base.clone(),
            type_index: RwLock::new(HashMap::new()),
            slot_mirror: OnceLock::new(),
            invalid_type: self.invalid_type,
        }
    }
}

impl std::fmt::Debug for FrozenGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frozen({} nodes, {} edges)",
            self.node_count, self.edge_count
        )
    }
}

#[cfg(test)]
impl FrozenGraph {
    /// The frozen form of a heap graph whose nodes are all column-backed
    /// records: every slot, vacancy and edge slot kept, each node's run sorted
    /// by type ordinal then slot descending, with the type mask. What a save
    /// of that graph and a mapped load of the file produce, built in memory so
    /// a randomised test needs no file.
    pub(crate) fn from_heap(
        graph: &petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
    ) -> Self {
        use petgraph::visit::{EdgeIndexable, NodeIndexable};
        let (node_bound, edge_bound) = (graph.node_bound(), graph.edge_bound());
        let mut nodes = Vec::with_capacity(node_bound);
        let mut words = vec![0u64; node_bound.div_ceil(64)];
        let mut node_count = 0;
        for slot in 0..node_bound {
            match graph.node_weight(NodeIndex::new(slot)) {
                Some(node) => {
                    nodes.push(node.clone());
                    words[slot >> 6] |= 1 << (slot & 63);
                    node_count += 1;
                }
                None => nodes.push(promote::placeholder_node()),
            }
        }
        let mut types: Vec<InternedKey> = graph
            .edge_weights()
            .map(|weight| weight.connection_type)
            .collect();
        types.sort();
        types.dedup();
        let ordinal = |key: InternedKey| types.binary_search(&key).expect("collected") as u16;
        let vacant = FrozenEndpoint {
            src: VACANT,
            dst: 0,
            ty: 0,
            pad: 0,
        };
        let mut endpoints = vec![vacant; edge_bound];
        let mut edge_props = FxHashMap::default();
        let mut edge_count = 0;
        for edge in petgraph::visit::IntoEdgeReferences::edge_references(graph) {
            use petgraph::visit::EdgeRef;
            let slot = edge.id().index();
            endpoints[slot] = FrozenEndpoint {
                src: edge.source().index() as u32,
                dst: edge.target().index() as u32,
                ty: ordinal(edge.weight().connection_type),
                pad: 0,
            };
            if !edge.weight().properties.is_empty() {
                edge_props.insert(slot as u32, edge.weight().clone());
            }
            edge_count += 1;
        }
        let csr = |outgoing: bool| {
            let mut offsets = vec![0u32; node_bound + 1];
            let (mut edges, mut tys) = (Vec::new(), Vec::new());
            for node in 0..node_bound {
                let mut run: Vec<(u16, std::cmp::Reverse<u32>, u32)> = endpoints
                    .iter()
                    .enumerate()
                    .filter(|(_, ep)| {
                        ep.src != VACANT
                            && (if outgoing { ep.src } else { ep.dst }) as usize == node
                    })
                    .map(|(slot, ep)| {
                        let peer = if outgoing { ep.dst } else { ep.src };
                        (ep.ty, std::cmp::Reverse(slot as u32), peer)
                    })
                    .collect();
                run.sort();
                for (ty, std::cmp::Reverse(slot), peer) in run {
                    edges.push(CsrEdge {
                        peer,
                        edge_idx: slot,
                    });
                    tys.push(ty);
                }
                offsets[node + 1] = edges.len() as u32;
            }
            (offsets, edges, tys)
        };
        let (out_off, out_edges, out_ty) = csr(true);
        let (in_off, in_edges, in_ty) = csr(false);
        let mut mask = vec![0u32; 2 * node_bound];
        for (side, (off, tys)) in [(&out_off, &out_ty), (&in_off, &in_ty)]
            .into_iter()
            .enumerate()
        {
            for node in 0..node_bound {
                for &ty in &tys[off[node] as usize..off[node + 1] as usize] {
                    mask[2 * node + side] |= 1 << (ty & 31);
                }
            }
        }
        FrozenGraph::new(FrozenParts {
            nodes: NodeSlab::owned(nodes),
            node_count,
            node_alive: (node_count != node_bound).then(|| FrozenArray::from_vec(words)),
            out_off: FrozenArray::from_vec(out_off),
            in_off: FrozenArray::from_vec(in_off),
            out_edges: FrozenArray::from_vec(out_edges),
            in_edges: FrozenArray::from_vec(in_edges),
            out_ty: FrozenArray::from_vec(out_ty),
            in_ty: FrozenArray::from_vec(in_ty),
            endpoints: FrozenArray::from_vec(endpoints),
            type_mask: Some(FrozenArray::from_vec(mask)),
            edge_count,
            types,
            edge_props,
        })
        .expect("a heap graph converts to a well-formed frozen graph")
    }

    /// This graph with its node records replaced, to build a topology whose
    /// records disagree with the rest of a loaded graph.
    pub(crate) fn with_nodes(&self, nodes: Vec<NodeData>) -> FrozenGraph {
        let mut changed = self.clone();
        changed.nodes = NodeSlab::owned(nodes);
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::property_storage::{ColumnarRow, PropertyStorage};

    fn node(ty: &str, row: u32) -> NodeData {
        NodeData {
            node_type: InternedKey::from_str(ty),
            ident: None,
            properties: PropertyStorage::Columnar(ColumnarRow::new(row)),
        }
    }

    /// Three nodes (slot 1 vacant), edges 0 -> 2 (type 0) and 2 -> 2 (type 1).
    fn parts() -> FrozenParts {
        let edge = |peer, edge_idx| CsrEdge { peer, edge_idx };
        FrozenParts {
            nodes: NodeSlab::owned(vec![node("A", 0), node("A", 1), node("B", 0)]),
            node_count: 2,
            node_alive: Some(FrozenArray::from_vec(vec![0b101])),
            out_off: FrozenArray::from_vec(vec![0, 1, 1, 2]),
            in_off: FrozenArray::from_vec(vec![0, 0, 0, 2]),
            out_edges: FrozenArray::from_vec(vec![edge(2, 0), edge(2, 1)]),
            in_edges: FrozenArray::from_vec(vec![edge(0, 0), edge(2, 1)]),
            out_ty: FrozenArray::from_vec(vec![0, 1]),
            in_ty: FrozenArray::from_vec(vec![0, 1]),
            endpoints: FrozenArray::from_vec(vec![
                FrozenEndpoint {
                    src: 0,
                    dst: 2,
                    ty: 0,
                    pad: 0,
                },
                FrozenEndpoint {
                    src: 2,
                    dst: 2,
                    ty: 1,
                    pad: 0,
                },
            ]),
            type_mask: None,
            edge_count: 2,
            types: vec![InternedKey::from_str("E1"), InternedKey::from_str("E2")]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            edge_props: FxHashMap::default(),
        }
    }

    #[test]
    fn well_formed_parts_build_and_answer() {
        let graph = FrozenGraph::new(parts()).expect("well-formed");
        assert_eq!(graph.live_node_count(), 2);
        assert!(graph.node_is_alive(0) && !graph.node_is_alive(1) && graph.node_is_alive(2));
        assert!(!graph.node_is_alive(3), "past the bound is vacant");
        assert_eq!(graph.run(2, Direction::Incoming), (0, 2));
        let ordinal = graph.ordinal_of(InternedKey::from_str("E1")).unwrap();
        let (lo, hi) = graph.typed_run(2, Direction::Incoming, ordinal);
        assert_eq!(hi - lo, 1, "the typed run holds only that type's entry");
    }

    #[test]
    fn every_inconsistency_between_the_arrays_is_refused() {
        type Mutation = Box<dyn Fn(&mut FrozenParts)>;
        let cases: Vec<(&str, Mutation)> = vec![
            (
                "short offsets",
                Box::new(|p| p.out_off = FrozenArray::from_vec(vec![0, 1, 2])),
            ),
            (
                "offsets not ending at the edge count",
                Box::new(|p| p.in_off = FrozenArray::from_vec(vec![0, 0, 0, 1])),
            ),
            (
                "offsets not monotone",
                Box::new(|p| p.out_off = FrozenArray::from_vec(vec![0, 2, 1, 2])),
            ),
            (
                "adjacency shorter than the edge count",
                Box::new(|p| p.out_ty = FrozenArray::from_vec(vec![0])),
            ),
            ("more edges than slots", Box::new(|p| p.edge_count = 3)),
            ("types out of order", Box::new(|p| p.types.reverse())),
            (
                "duplicate types",
                Box::new(|p| p.types = vec![p.types[0], p.types[0]]),
            ),
            (
                "alive bitset popcount",
                Box::new(|p| p.node_alive = Some(FrozenArray::from_vec(vec![0b111]))),
            ),
            (
                "alive bit past the bound",
                Box::new(|p| p.node_alive = Some(FrozenArray::from_vec(vec![0b1001]))),
            ),
            (
                "alive bitset too short",
                Box::new(|p| p.node_alive = Some(FrozenArray::from_vec(vec![]))),
            ),
            (
                "vacant nodes without a bitset",
                Box::new(|p| p.node_alive = None),
            ),
            ("edges without types", Box::new(|p| p.types = Vec::new())),
        ];
        for (what, mutate) in cases {
            let mut broken = parts();
            mutate(&mut broken);
            assert!(FrozenGraph::new(broken).is_err(), "{what} must be refused");
        }
    }

    /// The shape of a crafted file that lists no connection types and counts no
    /// edges but carries live endpoint records.
    fn typeless_parts() -> FrozenParts {
        let mut p = parts();
        p.out_off = FrozenArray::from_vec(vec![0, 0, 0, 0]);
        p.in_off = FrozenArray::from_vec(vec![0, 0, 0, 0]);
        p.out_edges = FrozenArray::from_vec(vec![]);
        p.in_edges = FrozenArray::from_vec(vec![]);
        p.out_ty = FrozenArray::from_vec(vec![]);
        p.in_ty = FrozenArray::from_vec(vec![]);
        p.edge_count = 0;
        p.types = Vec::new();
        p
    }

    #[test]
    fn a_type_ordinal_the_file_does_not_list_reads_without_panicking() {
        let graph = FrozenGraph::new(typeless_parts()).expect("every check passes");
        for ty in [0, 1, u16::MAX] {
            assert!(graph.edge_data(0, ty).properties.is_empty());
            assert_eq!(graph.type_key(ty), InternedKey::default());
        }
        let mut seen = 0;
        graph.for_each_endpoint_key(|_, _, _| seen += 1);
        assert_eq!(seen, 2, "the endpoint records are still walked");
        let listed = FrozenGraph::new(parts()).unwrap();
        assert_eq!(
            listed.edge_data(0, 7).connection_type,
            InternedKey::from_str("E1"),
            "an ordinal past the list borrows the first template"
        );
    }
}
