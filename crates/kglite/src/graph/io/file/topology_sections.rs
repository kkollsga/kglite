//! Raw topology in a v9 container's data region: the sections a mapped load
//! serves a [`FrozenGraph`] from.
//!
//! The compressed `topology` section stays in every file (a memory load and an
//! older reader decode it); these sections duplicate the same graph in a form
//! that needs no decode. Owner `topology`, little-endian, one array per section:
//!
//! - `topo:meta` (kind 10): JSON, the slot bounds, live counts and the node and
//!   connection type names (so a load can intern them without reading a node).
//! - `topo:nodes` (kind 11): 32-byte `NodeData` records, one per node slot; see
//!   `storage::frozen::node_record` for the layout.
//! - `topo:alive` (kind 12): `u64` bitset of occupied node slots; absent when
//!   every slot is occupied.
//! - `topo:out_off` / `topo:in_off` (kind 13): `u32`, `node_bound + 1` offsets.
//! - `topo:out_edges` / `topo:in_edges` (kind 14): `(peer, edge slot)` `u32`
//!   pairs, each node's run sorted by connection-type ordinal then edge slot
//!   descending.
//! - `topo:out_ty` / `topo:in_ty` (kind 15): `u16` connection-type ordinal of
//!   each adjacency entry.
//! - `topo:edges` (kind 16): `(src, dst, ordinal, 0)` per edge slot, 12 bytes;
//!   `src == u32::MAX` marks a vacant slot.
//! - `topo:type_mask` (kind 18): `u32`, `[out, in]` per node; bit `ordinal % 32`
//!   is set when the node has an edge of that type in that direction. Optional:
//!   a file without it is served by the run lookup alone.
//! - `topo:edge_props` (kind 19): codec-encoded `(edge slot, EdgeData)` pairs
//!   for the edges that carry properties, zstd-compressed (level 3); absent when
//!   none do. Kind 17 is the same payload uncompressed, which a load still reads.
//!   Either way the load decodes it to a heap map: properties are not mapped.
//!
//! A save writes them for a graph of at least the index floors (nodes or edges)
//! whose every node is a column-backed node without an inline identity, whose
//! rows follow node order, and that holds at most 65,535 node types and as many
//! connection types (the `u16` ordinal), and at most 262,144 relationships with
//! properties. Anything else stays on the compressed
//! topology alone, and the reason is recorded in the file. A load into mapped storage that
//! finds a complete, well-formed set serves the graph from it and never reads
//! the compressed section; a memory load never reads these.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::mappable::{
    MapEntry, MappableSection, MappableTable, MappedSection, Placement, SectionDtype,
};
use super::*;
use crate::graph::dir_graph::mapped_report::PROPERTY_EDGE_REASON;
use crate::graph::dir_graph::MappedFallback;
use crate::graph::schema::{EdgeData, InternedKey};
use crate::graph::storage::disk::csr::CsrEdge;
use crate::graph::storage::frozen::array::FrozenArray;
use crate::graph::storage::frozen::node_record::{self, NodeSlab};
use crate::graph::storage::frozen::{FrozenEndpoint, FrozenGraph, FrozenParts, VACANT};
use crate::graph::storage::interner::StringInterner;
use petgraph::graph::NodeIndex;

pub(super) const OWNER: &str = "topology";

pub(super) const KIND_META: u8 = 10;
pub(super) const KIND_NODES: u8 = 11;
pub(super) const KIND_ALIVE: u8 = 12;
pub(super) const KIND_OFFSETS: u8 = 13;
pub(super) const KIND_ADJACENCY: u8 = 14;
pub(super) const KIND_ADJACENCY_TYPES: u8 = 15;
pub(super) const KIND_ENDPOINTS: u8 = 16;
/// Edge properties, uncompressed: the form written before the section was compressed.
pub(super) const KIND_EDGE_PROPS: u8 = 17;
pub(super) const KIND_TYPE_MASK: u8 = 18;
pub(super) const KIND_EDGE_PROPS_ZSTD: u8 = 19;

const META: &str = "topo:meta";
const NODES: &str = "topo:nodes";
const ALIVE: &str = "topo:alive";
const OUT_OFF: &str = "topo:out_off";
const IN_OFF: &str = "topo:in_off";
const OUT_EDGES: &str = "topo:out_edges";
const IN_EDGES: &str = "topo:in_edges";
const OUT_TY: &str = "topo:out_ty";
const IN_TY: &str = "topo:in_ty";
const EDGES: &str = "topo:edges";
const TYPE_MASK: &str = "topo:type_mask";
const EDGE_PROPS: &str = "topo:edge_props";

/// The name a record with an out-of-range connection type ordinal reads as, so
/// resolving the key of any edge of a crafted file yields a name, not a panic.
const INVALID_TYPE: &str = "<invalid connection type>";

/// Version of the section set; a reader refuses a newer one by ignoring it.
const VERSION: u32 = 1;

/// Types (node or connection) beyond which a save records nothing: a
/// connection type is a `u16` ordinal in the adjacency arrays.
const MAX_TYPES: usize = u16::MAX as usize;

/// Relationships with properties beyond which a save records no topology
/// sections. Their properties are decoded to a heap map at load, so past this a
/// file triples in size (234 to 757 MB at 12M relationships with one float
/// property) for a 4 % smaller footprint until the properties have mapped
/// columns.
const MAX_PROPERTY_EDGES: usize = 262_144;

#[derive(Serialize, Deserialize)]
struct Meta {
    version: u32,
    node_bound: u64,
    edge_bound: u64,
    node_count: u64,
    edge_count: u64,
    node_types: Vec<String>,
    /// Connection type names in ordinal order (ascending by `InternedKey`).
    conn_types: Vec<String>,
}

fn owned(
    name: &str,
    kind: u8,
    dtype: SectionDtype,
    count: usize,
    data: Vec<u8>,
) -> MappableSection<'static> {
    MappableSection {
        name: name.to_string(),
        owner: OWNER.to_string(),
        kind,
        dtype,
        elem_count: count as u64,
        flags: 0,
        data: std::borrow::Cow::Owned(data),
    }
}

fn le_u32(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn le_u16(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn le_u64(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The node half of a save: records, vacancy bitset and type names, or `None`
/// when some node cannot be a record (inline identity, staged properties, a
/// row that is not its rank within its type).
struct NodeSide {
    records: Vec<u8>,
    alive: Option<Vec<u64>>,
    node_count: usize,
    type_names: Vec<String>,
}

fn collect_nodes(graph: &DirGraph) -> Result<NodeSide, &'static str> {
    let bound = graph.graph.node_bound();
    let mut records = Vec::with_capacity(bound * node_record::RECORD_BYTES);
    let mut next_row: FxHashMap<InternedKey, u32> = FxHashMap::default();
    let mut words = vec![0u64; bound.div_ceil(64)];
    let mut live = 0usize;
    for slot in 0..bound {
        match graph.graph.node_weight(NodeIndex::new(slot)) {
            None => node_record::encode_vacant(&mut records),
            Some(node) => {
                let row = node
                    .properties
                    .columnar_row_id()
                    .ok_or("a node's properties are not column-backed")?;
                if node.ident.is_some() {
                    return Err("a node holds an inline identity");
                }
                let expected = next_row.entry(node.node_type).or_insert(0);
                if row != *expected {
                    return Err("a node's column row is not its rank within its type");
                }
                *expected += 1;
                node_record::encode_record(&mut records, node.node_type, row);
                words[slot >> 6] |= 1 << (slot & 63);
                live += 1;
            }
        }
    }
    if next_row.len() > max_types() {
        return Err("more than 65,535 node types");
    }
    let mut type_names = Vec::with_capacity(next_row.len());
    for key in next_row.keys() {
        let name = graph
            .interner
            .try_resolve(*key)
            .ok_or("a node type has no interned name")?;
        type_names.push(name.to_string());
    }
    type_names.sort();
    Ok(NodeSide {
        records,
        alive: (live != bound).then_some(words),
        node_count: live,
        type_names,
    })
}

/// The edge half: adjacency arrays and endpoint records, the connection type
/// names in ordinal order, and the edges that carry properties.
struct EdgeSide {
    edge_bound: usize,
    edge_count: usize,
    endpoints: Vec<FrozenEndpoint>,
    out_off: Vec<u32>,
    in_off: Vec<u32>,
    out_edges: Vec<CsrEdge>,
    in_edges: Vec<CsrEdge>,
    out_ty: Vec<u16>,
    in_ty: Vec<u16>,
    conn_names: Vec<String>,
    props: Vec<(u32, EdgeData)>,
}

/// Counting-sort the live edges into one direction's CSR: each node's run holds
/// its edges by edge slot descending, then (when the graph has several types)
/// stably regrouped by ordinal.
fn build_csr(
    endpoints: &[FrozenEndpoint],
    node_bound: usize,
    edge_count: usize,
    outgoing: bool,
    multi_type: bool,
) -> (Vec<u32>, Vec<CsrEdge>, Vec<u16>) {
    let node_of = |ep: &FrozenEndpoint| if outgoing { ep.src } else { ep.dst } as usize;
    let mut offsets = vec![0u32; node_bound + 1];
    for ep in endpoints.iter().filter(|ep| ep.src != VACANT) {
        offsets[node_of(ep) + 1] += 1;
    }
    for i in 0..node_bound {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets[..node_bound].to_vec();
    let mut edges = vec![CsrEdge::default(); edge_count];
    let mut types = vec![0u16; edge_count];
    for slot in (0..endpoints.len()).rev() {
        let ep = endpoints[slot];
        if ep.src == VACANT {
            continue;
        }
        let node = node_of(&ep);
        let at = cursor[node] as usize;
        cursor[node] += 1;
        edges[at] = CsrEdge {
            peer: if outgoing { ep.dst } else { ep.src },
            edge_idx: slot as u32,
        };
        types[at] = ep.ty;
    }
    if multi_type {
        for node in 0..node_bound {
            let (lo, hi) = (offsets[node] as usize, offsets[node + 1] as usize);
            if hi - lo < 2 || types[lo..hi].windows(2).all(|w| w[0] == w[1]) {
                continue;
            }
            let mut run: Vec<(u16, CsrEdge)> = (lo..hi).map(|i| (types[i], edges[i])).collect();
            run.sort_by_key(|(ty, _)| *ty);
            for (i, (ty, edge)) in run.into_iter().enumerate() {
                types[lo + i] = ty;
                edges[lo + i] = edge;
            }
        }
    }
    (offsets, edges, types)
}

fn collect_edges(graph: &DirGraph) -> Result<EdgeSide, &'static str> {
    let node_bound = graph.graph.node_bound();
    let edge_bound = graph.graph.edge_bound();
    let mut keys: FxHashSet<InternedKey> = FxHashSet::default();
    let mut edge_count = 0usize;
    let mut property_edges = 0usize;
    let mut last: Option<InternedKey> = None;
    for edge in graph.graph.edge_references() {
        let key = edge.connection_type();
        if last != Some(key) {
            keys.insert(key);
            last = Some(key);
        }
        if !edge.weight().properties.is_empty() {
            property_edges += 1;
        }
        edge_count += 1;
    }
    if keys.len() > max_types() {
        return Err("more than 65,535 connection types");
    }
    if property_edges > MAX_PROPERTY_EDGES {
        return Err(PROPERTY_EDGE_REASON);
    }
    let mut ordered: Vec<InternedKey> = keys.into_iter().collect();
    ordered.sort();
    let conn_names = ordered
        .iter()
        .map(|key| graph.interner.try_resolve(*key).map(str::to_string))
        .collect::<Option<Vec<_>>>()
        .ok_or("a connection type has no interned name")?;
    let ordinal: FxHashMap<InternedKey, u16> = ordered
        .iter()
        .enumerate()
        .map(|(i, key)| (*key, i as u16))
        .collect();
    let mut endpoints = vec![
        FrozenEndpoint {
            src: VACANT,
            dst: 0,
            ty: 0,
            pad: 0
        };
        edge_bound
    ];
    let mut props = Vec::with_capacity(property_edges);
    for edge in graph.graph.edge_references() {
        let slot = edge.id().index();
        let (src, dst) = (edge.source().index(), edge.target().index());
        if src >= node_bound || dst >= node_bound || slot >= edge_bound {
            return Err("an edge endpoint lies beyond the node array");
        }
        endpoints[slot] = FrozenEndpoint {
            src: src as u32,
            dst: dst as u32,
            ty: *ordinal
                .get(&edge.connection_type())
                .ok_or("a connection type has no ordinal")?,
            pad: 0,
        };
        let weight = edge.weight();
        if !weight.properties.is_empty() {
            props.push((slot as u32, weight.clone()));
        }
    }
    props.sort_by_key(|(slot, _)| *slot);
    let multi_type = ordered.len() > 1;
    let (out_off, out_edges, out_ty) =
        build_csr(&endpoints, node_bound, edge_count, true, multi_type);
    let (in_off, in_edges, in_ty) =
        build_csr(&endpoints, node_bound, edge_count, false, multi_type);
    Ok(EdgeSide {
        edge_bound,
        edge_count,
        endpoints,
        out_off,
        in_off,
        out_edges,
        in_edges,
        out_ty,
        in_ty,
        conn_names,
        props,
    })
}

/// The presence masks of a save, `[out, in]` per node: bit `ordinal % 32` of
/// each node's run, so a run of at most 32 types is recorded exactly.
fn type_masks(out_off: &[u32], out_ty: &[u16], in_off: &[u32], in_ty: &[u16]) -> Vec<u32> {
    let bound = out_off.len().saturating_sub(1);
    let mut masks = vec![0u32; 2 * bound];
    for (side, (off, tys)) in [(out_off, out_ty), (in_off, in_ty)].into_iter().enumerate() {
        for node in 0..bound {
            let run = &tys[off[node] as usize..off[node + 1] as usize];
            masks[2 * node + side] = run.iter().fold(0u32, |m, &t| m | 1 << (t & 31));
        }
    }
    masks
}

fn csr_bytes(edges: &[CsrEdge]) -> Vec<u8> {
    let mut out = Vec::with_capacity(edges.len() * 8);
    for edge in edges {
        out.extend_from_slice(&edge.peer.to_le_bytes());
        out.extend_from_slice(&edge.edge_idx.to_le_bytes());
    }
    out
}

fn endpoint_bytes(endpoints: &[FrozenEndpoint]) -> Vec<u8> {
    let mut out = Vec::with_capacity(endpoints.len() * 12);
    for ep in endpoints {
        out.extend_from_slice(&ep.src.to_le_bytes());
        out.extend_from_slice(&ep.dst.to_le_bytes());
        out.extend_from_slice(&ep.ty.to_le_bytes());
        out.extend_from_slice(&ep.pad.to_le_bytes());
    }
    out
}

#[cfg(test)]
thread_local! {
    static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static NO_TYPE_MASK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static RAW_EDGE_PROPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TYPE_CAP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Run `body` with the type cap a save applies replaced, so a test can reach
/// the refusal without building 65,536 types.
#[cfg(test)]
pub(crate) fn with_type_cap<R>(cap: usize, body: impl FnOnce() -> R) -> R {
    TYPE_CAP.with(|flag| flag.set(Some(cap)));
    let result = body();
    TYPE_CAP.with(|flag| flag.set(None));
    result
}

fn max_types() -> usize {
    #[cfg(test)]
    if let Some(cap) = TYPE_CAP.with(|flag| flag.get()) {
        return cap;
    }
    MAX_TYPES
}

/// Run `body` with the edge-property section written uncompressed (kind 17),
/// the shape of a v9 file written before it was compressed.
#[cfg(test)]
pub(crate) fn with_raw_edge_props<R>(body: impl FnOnce() -> R) -> R {
    RAW_EDGE_PROPS.with(|flag| flag.set(true));
    let result = body();
    RAW_EDGE_PROPS.with(|flag| flag.set(false));
    result
}

/// Run `body` with the type-mask section left out of a save, the shape of a v9
/// file written before it existed.
#[cfg(test)]
pub(crate) fn without_type_mask<R>(body: impl FnOnce() -> R) -> R {
    NO_TYPE_MASK.with(|flag| flag.set(true));
    let result = body();
    NO_TYPE_MASK.with(|flag| flag.set(false));
    result
}

/// Run `body` with topology sections switched off, so a test can save a file
/// whose mapped load keeps a heap topology (the shape every older v9 file, and a
/// graph that cannot be frozen, still loads as).
#[cfg(test)]
pub(crate) fn without_topology<R>(body: impl FnOnce() -> R) -> R {
    DISABLED.with(|flag| flag.set(true));
    let result = body();
    DISABLED.with(|flag| flag.set(false));
    result
}

/// The topology sections a save of `graph` writes, and the reason it wrote
/// none when a graph at or above the floors could not be recorded. A graph
/// below the floors, a big-endian host and a layout that asks for none are not
/// reasons: nothing was dropped.
pub(super) fn collect_topology_sections(
    graph: &DirGraph,
) -> io::Result<(Vec<MappableSection<'static>>, Option<MappedFallback>)> {
    let none = |reason: Option<&str>| {
        Ok((
            Vec::new(),
            reason.map(|reason| MappedFallback::new("topology", reason)),
        ))
    };
    if !cfg!(target_endian = "little") {
        return none(None);
    }
    #[cfg(test)]
    if DISABLED.with(|flag| flag.get()) {
        return none(None);
    }
    let (min_nodes, min_edges) = super::index_sections::floors();
    let (nodes, edges) = (graph.graph.node_count(), graph.graph.edge_count());
    if nodes < min_nodes && edges < min_edges {
        return none(None);
    }
    if graph.graph.node_bound() >= u32::MAX as usize
        || graph.graph.edge_bound() >= u32::MAX as usize
    {
        return none(Some("node or edge slots beyond 4,294,967,294"));
    }
    let node_side = match collect_nodes(graph) {
        Ok(side) => side,
        Err(reason) => return none(Some(reason)),
    };
    let edge_side = match collect_edges(graph) {
        Ok(side) => side,
        Err(reason) => return none(Some(reason)),
    };
    let meta = Meta {
        version: VERSION,
        node_bound: graph.graph.node_bound() as u64,
        edge_bound: edge_side.edge_bound as u64,
        node_count: node_side.node_count as u64,
        edge_count: edge_side.edge_count as u64,
        node_types: node_side.type_names,
        conn_types: edge_side.conn_names,
    };
    let mut out = vec![owned(
        META,
        KIND_META,
        SectionDtype::Raw,
        0,
        serde_json::to_vec(&meta).map_err(io::Error::other)?,
    )];
    let meta_len = out[0].data.len();
    out[0].elem_count = meta_len as u64;
    let node_words = node_side.records.len() / 8;
    out.push(owned(
        NODES,
        KIND_NODES,
        SectionDtype::U64,
        node_words,
        node_side.records,
    ));
    if let Some(words) = &node_side.alive {
        out.push(owned(
            ALIVE,
            KIND_ALIVE,
            SectionDtype::U64,
            words.len(),
            le_u64(words),
        ));
    }
    for (name, offsets) in [(OUT_OFF, &edge_side.out_off), (IN_OFF, &edge_side.in_off)] {
        out.push(owned(
            name,
            KIND_OFFSETS,
            SectionDtype::U32,
            offsets.len(),
            le_u32(offsets),
        ));
    }
    for (name, edges) in [
        (OUT_EDGES, &edge_side.out_edges),
        (IN_EDGES, &edge_side.in_edges),
    ] {
        out.push(owned(
            name,
            KIND_ADJACENCY,
            SectionDtype::U32,
            edges.len() * 2,
            csr_bytes(edges),
        ));
    }
    for (name, tys) in [(OUT_TY, &edge_side.out_ty), (IN_TY, &edge_side.in_ty)] {
        out.push(owned(
            name,
            KIND_ADJACENCY_TYPES,
            SectionDtype::U16,
            tys.len(),
            le_u16(tys),
        ));
    }
    out.push(owned(
        EDGES,
        KIND_ENDPOINTS,
        SectionDtype::U32,
        edge_side.endpoints.len() * 3,
        endpoint_bytes(&edge_side.endpoints),
    ));
    #[cfg(test)]
    let write_mask = !NO_TYPE_MASK.with(|flag| flag.get());
    #[cfg(not(test))]
    let write_mask = true;
    if write_mask {
        let masks = type_masks(
            &edge_side.out_off,
            &edge_side.out_ty,
            &edge_side.in_off,
            &edge_side.in_ty,
        );
        out.push(owned(
            TYPE_MASK,
            KIND_TYPE_MASK,
            SectionDtype::U32,
            masks.len(),
            le_u32(&masks),
        ));
    }
    if !edge_side.props.is_empty() {
        let encoded = {
            let _guard = SerdeSerializeGuard::new(&graph.interner);
            codec_ser(serde_codec::CodecVersion::PostcardV1, &edge_side.props)?
        };
        #[cfg(test)]
        let raw = RAW_EDGE_PROPS.with(|flag| flag.get());
        #[cfg(not(test))]
        let raw = false;
        let (kind, payload) = if raw {
            (KIND_EDGE_PROPS, encoded)
        } else {
            (
                KIND_EDGE_PROPS_ZSTD,
                zstd::encode_all(encoded.as_slice(), 3)?,
            )
        };
        out.push(owned(
            EDGE_PROPS,
            kind,
            SectionDtype::Raw,
            payload.len(),
            payload,
        ));
    }
    Ok((out, None))
}

/// Where a load puts one topology section: the rule the index sections follow
/// (mapped into mapped storage, otherwise never read).
pub(super) fn place(
    entry: &MapEntry,
    file: Option<&Arc<mappable::MappedFile>>,
    policy: load_options::MapPolicy,
    into_mapped: bool,
) -> Placement {
    super::index_sections::place(entry, file, policy, into_mapped)
}

/// The topology sections of an opened table, when it holds a complete set.
pub(super) struct TopologySections {
    meta: MappedSection,
    nodes: MappedSection,
    alive: Option<MappedSection>,
    out_off: MappedSection,
    in_off: MappedSection,
    out_edges: MappedSection,
    in_edges: MappedSection,
    out_ty: MappedSection,
    in_ty: MappedSection,
    endpoints: MappedSection,
    type_mask: Option<MappedSection>,
    /// The section and whether it is zstd-compressed (kind 19) or raw (kind 17).
    edge_props: Option<(MappedSection, bool)>,
}

impl TopologySections {
    /// `None` unless every required section is present with the kind and
    /// element type this reader expects: a partial or foreign set is ignored,
    /// and the load decodes the compressed topology as it always did.
    pub(super) fn from_table(table: &MappableTable) -> Option<Self> {
        let get = |name: &str, kind: u8, dtype: SectionDtype| {
            table
                .get(name)
                .filter(|s| s.owner == OWNER && s.kind == kind && s.dtype == dtype)
                .cloned()
        };
        Some(TopologySections {
            meta: get(META, KIND_META, SectionDtype::Raw)?,
            nodes: get(NODES, KIND_NODES, SectionDtype::U64)?,
            alive: get(ALIVE, KIND_ALIVE, SectionDtype::U64),
            out_off: get(OUT_OFF, KIND_OFFSETS, SectionDtype::U32)?,
            in_off: get(IN_OFF, KIND_OFFSETS, SectionDtype::U32)?,
            out_edges: get(OUT_EDGES, KIND_ADJACENCY, SectionDtype::U32)?,
            in_edges: get(IN_EDGES, KIND_ADJACENCY, SectionDtype::U32)?,
            out_ty: get(OUT_TY, KIND_ADJACENCY_TYPES, SectionDtype::U16)?,
            in_ty: get(IN_TY, KIND_ADJACENCY_TYPES, SectionDtype::U16)?,
            endpoints: get(EDGES, KIND_ENDPOINTS, SectionDtype::U32)?,
            type_mask: get(TYPE_MASK, KIND_TYPE_MASK, SectionDtype::U32),
            edge_props: get(EDGE_PROPS, KIND_EDGE_PROPS_ZSTD, SectionDtype::Raw)
                .map(|section| (section, true))
                .or_else(|| {
                    get(EDGE_PROPS, KIND_EDGE_PROPS, SectionDtype::Raw)
                        .map(|section| (section, false))
                }),
        })
    }

    /// Validate the sections and build the topology they describe, interning
    /// every type and edge-property name into `interner`.
    pub(super) fn build(
        self,
        codec: serde_codec::CodecVersion,
        interner: &mut StringInterner,
        ceiling: Option<u64>,
    ) -> io::Result<FrozenGraph> {
        self.meta.verify()?;
        let meta: Meta = serde_json::from_slice(self.meta.as_bytes())
            .map_err(|e| invalid_data(format!("topology metadata is unreadable: {e}")))?;
        if meta.version != VERSION {
            return Err(invalid_data("topology sections have an unknown version"));
        }
        for name in meta.node_types.iter().chain(&meta.conn_types) {
            interner
                .try_get_or_intern(name)
                .map_err(|e| invalid_data(e.to_string()))?;
        }
        let nodes = NodeSlab::from_section(&self.nodes)?;
        let shaped = |what: &str| invalid_data(format!("topology section {what} is malformed"));
        let node_alive = match &self.alive {
            Some(section) => {
                Some(FrozenArray::<u64>::from_section(section).ok_or_else(|| shaped(ALIVE))?)
            }
            None => None,
        };
        let types: Vec<InternedKey> = meta
            .conn_types
            .iter()
            .map(|name| InternedKey::from_str(name))
            .collect();
        let edge_props = match &self.edge_props {
            Some((section, compressed)) => {
                section.verify()?;
                let inflated;
                let bytes = if *compressed {
                    inflated = zstd_decompress_within(section.as_bytes(), ceiling)?;
                    inflated.as_slice()
                } else {
                    section.as_bytes()
                };
                let _guard = SerdeDeserializeGuard::new(interner);
                let decoded: Vec<(u32, EdgeData)> = codec_deser(codec, bytes, bytes.len() as u64)?;
                decoded.into_iter().collect::<FxHashMap<_, _>>()
            }
            None => FxHashMap::default(),
        };
        let parts = FrozenParts {
            nodes,
            node_count: usize::try_from(meta.node_count).map_err(|_| shaped(META))?,
            node_alive,
            out_off: FrozenArray::from_section(&self.out_off).ok_or_else(|| shaped(OUT_OFF))?,
            in_off: FrozenArray::from_section(&self.in_off).ok_or_else(|| shaped(IN_OFF))?,
            out_edges: FrozenArray::from_section(&self.out_edges)
                .ok_or_else(|| shaped(OUT_EDGES))?,
            in_edges: FrozenArray::from_section(&self.in_edges).ok_or_else(|| shaped(IN_EDGES))?,
            out_ty: FrozenArray::from_section(&self.out_ty).ok_or_else(|| shaped(OUT_TY))?,
            in_ty: FrozenArray::from_section(&self.in_ty).ok_or_else(|| shaped(IN_TY))?,
            endpoints: FrozenArray::from_section(&self.endpoints).ok_or_else(|| shaped(EDGES))?,
            type_mask: match &self.type_mask {
                Some(section) => {
                    Some(FrozenArray::from_section(section).ok_or_else(|| shaped(TYPE_MASK))?)
                }
                None => None,
            },
            edge_count: usize::try_from(meta.edge_count).map_err(|_| shaped(META))?,
            types,
            edge_props,
        };
        if parts.nodes.as_slice().len() as u64 != meta.node_bound
            || parts.endpoints.len() as u64 != meta.edge_bound
        {
            return Err(invalid_data(
                "topology metadata disagrees with the node or edge array lengths",
            ));
        }
        let mut frozen = FrozenGraph::new(parts)?;
        check_node_types(&frozen, &meta.node_types)?;
        let invalid_type = interner
            .try_get_or_intern(INVALID_TYPE)
            .map_err(|e| invalid_data(e.to_string()))?;
        frozen.set_invalid_type(invalid_type);
        Ok(frozen)
    }
}

/// Every live node record must name one of the `listed` types, which the load
/// has interned. A record naming another would read as a node type with no
/// name, so it is refused here whether or not the file's member lists are
/// served. One pass over records the load has already read.
fn check_node_types(frozen: &FrozenGraph, listed: &[String]) -> io::Result<()> {
    let known: FxHashSet<InternedKey> = listed.iter().map(|n| InternedKey::from_str(n)).collect();
    for (slot, node) in frozen.node_slice().iter().enumerate() {
        if frozen.node_is_alive(slot) && !known.contains(&node.node_type) {
            return Err(invalid_data(format!(
                "topology node record {slot} names a type that the topology metadata does not list"
            )));
        }
    }
    Ok(())
}

/// Check that a frozen graph's node records agree with the type indexes the
/// load installed: each type's members, in ascending order, must be records of
/// that type whose row is the member's rank. The column loader binds row `k`
/// of a type to its `k`-th member, so a record that says otherwise would read
/// another node's properties.
pub(super) fn check_rows_against_types(graph: &DirGraph) -> io::Result<()> {
    for (type_name, members) in graph.type_indices.iter() {
        let key = InternedKey::from_str(type_name);
        let mut previous: Option<usize> = None;
        for (rank, member) in members.iter().enumerate() {
            let ascending = previous.is_none_or(|p| member.index() > p);
            previous = Some(member.index());
            let row = graph
                .graph
                .node_weight(member)
                .filter(|node| node.node_type == key)
                .and_then(|node| node.properties.columnar_row_id());
            if !ascending || row != Some(rank as u32) {
                return Err(invalid_data(format!(
                    "frozen topology disagrees with the node index of type '{type_name}'"
                )));
            }
        }
    }
    Ok(())
}
