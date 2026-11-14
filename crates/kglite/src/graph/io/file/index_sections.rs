//! Derived indexes in a v9 container's data region: node type membership,
//! the int64 id index and per-edge-type peer counts.
//!
//! These are caches a load would otherwise rebuild from the topology (an
//! O(nodes) grouping pass, a heap hash map per id on first lookup, an
//! O(edges) scan on the first grouped aggregate). A save of a large graph
//! records them; a load into mapped storage serves them from the file instead
//! of the heap. A load into memory storage ignores them and builds exactly
//! what it always built.
//!
//! Sections (owner `indexes`, little-endian):
//! - `idx:type:<T>`: `u32` node indices of type `T`, ascending (kind 6).
//! - `idx:id:<T>:keys` / `idx:id:<T>:nodes`: the sorted unique ids of a type
//!   whose ids are all `UniqueId` (`u32` keys), all `Int64` (`i64` keys) or all
//!   strings (`u64` offsets then the key bytes, kind 19; see
//!   `id_index::string_keys`), and the `u32` node of each (kind 8). Any other
//!   id leaves the type to the lazy build.
//! - `idx:peer:<C>:out` / `idx:peer:<C>:in`: `(u32 node, u32 count)` pairs for
//!   connection type `C`, grouped by edge target and by edge source (kind 9).
//!
//! A section that fails its CRC32 or its structure checks is a damaged file
//! and refuses the load; a type section set whose member count disagrees with
//! the decoded topology is dropped and the indexes are rebuilt. The first edge
//! write drops the persisted peer counts; node writes layer over the id and
//! type indexes, as they do for a disk graph.

use rustc_hash::FxHashMap;

use super::embedding_sections::{escape, unescape};
use super::load_options::MapPolicy;
use super::mappable::{
    resolve_mapping, MapContext, MapEntry, MappableSection, MappableTable, MappedFile,
    MappedSection, Placement, SectionDtype,
};
use super::*;
use crate::datatypes::Value;
use crate::graph::dir_graph::MappedFallback;
use crate::graph::schema::InternedKey;
use crate::graph::storage::disk::id_index::{string_keys, IdIndexBase, IdIndexStore};
use crate::graph::storage::disk::type_index::{TypeIndexBase, TypeIndexStore};
use crate::graph::storage::mapped::peer_base::PeerBase;

pub(super) const OWNER: &str = "indexes";

/// Section classes (`MapEntry::kind`) owned by this module.
pub(super) const KIND_TYPE_MEMBERS: u8 = 6;
pub(super) const KIND_ID_KEYS: u8 = 7;
pub(super) const KIND_ID_NODES: u8 = 8;
pub(super) const KIND_PEERS: u8 = 9;
pub(super) const KIND_ID_STRINGS: u8 = 19;
/// The nodes of a type that share an id with a later node and so have no entry
/// in its id sections; written only for a type that has any.
pub(super) const KIND_ID_SHADOWED: u8 = 20;

/// A save records the type and id indexes of a graph with at least this many
/// live nodes. Below it the rebuild is too cheap to be worth file bytes, and
/// a small graph stays a v8 container older readers can open.
pub(crate) const MIN_NODES: usize = 65_536;

/// A save records the peer counts of a graph with at least this many live
/// edges, for the same reason.
pub(crate) const MIN_EDGES: usize = 262_144;

/// Types (node or connection) beyond which a save records nothing: each is a
/// few table rows, and a catalogue this wide is not what the sections are for.
const MAX_TYPES: usize = 1024;

#[cfg(test)]
thread_local! {
    static FLOORS: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
}

/// Run `body` with the node and edge floors a save applies replaced, so a
/// test can exercise the sections without building a quarter-million-edge graph.
#[cfg(test)]
pub(crate) fn with_floors<R>(nodes: usize, edges: usize, body: impl FnOnce() -> R) -> R {
    FLOORS.with(|floors| floors.set(Some((nodes, edges))));
    let result = body();
    FLOORS.with(|floors| floors.set(None));
    result
}

/// Debug builds only: `KGLITE_RAW_SECTION_FLOORS=<nodes>,<edges>` replaces the
/// floors, so a correctness run can record the index and topology sections for
/// graphs of any size and put every mapped load of a saved graph on them.
/// Release builds do not read it.
#[cfg(debug_assertions)]
const FLOORS_ENV_VAR: &str = "KGLITE_RAW_SECTION_FLOORS";

pub(super) fn floors() -> (usize, usize) {
    #[cfg(test)]
    if let Some(floors) = FLOORS.with(|floors| floors.get()) {
        return floors;
    }
    #[cfg(debug_assertions)]
    if let Some((nodes, edges)) = std::env::var(FLOORS_ENV_VAR)
        .ok()
        .as_deref()
        .and_then(|value| value.split_once(','))
        .and_then(|(n, e)| Some((n.trim().parse().ok()?, e.trim().parse().ok()?)))
    {
        return (nodes, edges);
    }
    (MIN_NODES, MIN_EDGES)
}

fn type_section_name(node_type: &str) -> String {
    format!("idx:type:{}", escape(node_type))
}

fn id_section_name(node_type: &str, part: &str) -> String {
    format!("idx:id:{}:{part}", escape(node_type))
}

fn peer_section_name(conn_type: &str, part: &str) -> String {
    format!("idx:peer:{}:{part}", escape(conn_type))
}

fn owned(
    name: String,
    kind: u8,
    dtype: SectionDtype,
    count: usize,
    data: Vec<u8>,
) -> MappableSection<'static> {
    MappableSection {
        name,
        owner: OWNER.to_string(),
        kind,
        dtype,
        elem_count: count as u64,
        flags: 0,
        data: std::borrow::Cow::Owned(data),
    }
}

/// The one id spelling a type's index can be written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IdKind {
    /// `Value::UniqueId`, what a DataFrame load stores: `u32` keys.
    Unique,
    /// `Value::Int64`, or a mix of it and `UniqueId`: `i64` keys.
    Int64,
    /// `Value::String`: byte-ordered keys, held in [`TypeScan::strings`].
    Str,
}

/// What one pass over a type's nodes collects.
#[derive(Default)]
struct TypeScan {
    members: Vec<u32>,
    /// `(id, node)` while every id seen so far has the same spelling.
    ids: Vec<(i64, u32)>,
    /// String ids: the bytes of every id seen, and `(start, len, node)` into them.
    strings: StringIds,
    kind: Option<IdKind>,
    /// False once an id of another spelling, or a node without one, is seen.
    uniform: bool,
    /// Why the type has no id section, once `uniform` is false.
    abandoned: Option<&'static str>,
}

#[derive(Default)]
struct StringIds {
    bytes: Vec<u8>,
    entries: Vec<(usize, u32, u32)>,
}

impl StringIds {
    fn push(&mut self, id: &str, node: u32) {
        let Ok(len) = u32::try_from(id.len()) else {
            return;
        };
        self.entries.push((self.bytes.len(), len, node));
        self.bytes.extend_from_slice(id.as_bytes());
    }

    fn key(&self, entry: &(usize, u32, u32)) -> &[u8] {
        &self.bytes[entry.0..entry.0 + entry.1 as usize]
    }
}

impl TypeScan {
    fn see(&mut self, id: &Value, node: u32) {
        let (kind, value) = match id {
            Value::UniqueId(id) => (IdKind::Unique, i64::from(*id)),
            Value::Int64(id) => (IdKind::Int64, *id),
            Value::String(id) if id.len() <= u32::MAX as usize => (IdKind::Str, 0),
            Value::String(_) => {
                self.abandon("a string id longer than 4 GiB");
                return;
            }
            Value::Null => {
                self.abandon("a node without an id");
                return;
            }
            _ => {
                self.abandon("ids that are neither integers nor strings");
                return;
            }
        };
        // `UniqueId` and `Int64` spell one integer: the lookup treats them as
        // equal, and a heap store widens its id column to `Int64` when the two
        // meet. A type holding both is written as `i64` keys.
        match (*self.kind.get_or_insert(kind), kind) {
            (held, kind) if held == kind => {}
            (IdKind::Unique | IdKind::Int64, IdKind::Unique | IdKind::Int64) => {
                self.kind = Some(IdKind::Int64);
            }
            _ => {
                self.abandon("ids of both integer and string kinds");
                return;
            }
        }
        if let Value::String(id) = id {
            self.strings.push(id, node);
        } else {
            self.ids.push((value, node));
        }
    }

    fn abandon(&mut self, reason: &'static str) {
        self.uniform = false;
        self.abandoned.get_or_insert(reason);
        self.ids = Vec::new();
        self.strings = StringIds::default();
    }
}

/// The index sections a save of `graph` writes, and the parts it left out of
/// them that a graph at or above the floors could have had: none for a small
/// graph or on a big-endian host (the bytes are the format's, not the host's).
pub(super) fn collect_index_sections(
    graph: &DirGraph,
) -> io::Result<(Vec<MappableSection<'static>>, Vec<MappedFallback>)> {
    let mut out = Vec::new();
    let mut fallbacks = Vec::new();
    if !cfg!(target_endian = "little") {
        return Ok((out, fallbacks));
    }
    let (min_nodes, min_edges) = floors();
    if graph.graph.node_count() >= min_nodes && graph.graph.node_bound() <= u32::MAX as usize {
        push_node_sections(graph, &mut out, &mut fallbacks);
    }
    if graph.graph.edge_count() >= min_edges && graph.graph.node_bound() <= u32::MAX as usize {
        push_peer_sections(graph, &mut out, &mut fallbacks);
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((out, fallbacks))
}

fn push_node_sections(
    graph: &DirGraph,
    out: &mut Vec<MappableSection<'static>>,
    fallbacks: &mut Vec<MappedFallback>,
) {
    let mut scans: FxHashMap<InternedKey, TypeScan> = FxHashMap::default();
    {
        let _guard = graph.graph.begin_query();
        for idx in graph.graph.node_indices() {
            let Some(node) = graph.graph.node_view(idx) else {
                continue;
            };
            let scan = scans.entry(node.node_type()).or_insert_with(|| TypeScan {
                uniform: true,
                ..TypeScan::default()
            });
            let position = idx.index() as u32;
            scan.members.push(position);
            if scan.uniform {
                scan.see(node.id().as_ref(), position);
            }
        }
    }
    if scans.len() > MAX_TYPES {
        fallbacks.push(MappedFallback::new(
            "indexes:nodes",
            "more than 1,024 node types: no member lists or id indexes are recorded",
        ));
        return;
    }
    for (type_key, mut scan) in scans {
        let Some(name) = graph.interner.try_resolve(type_key) else {
            fallbacks.push(MappedFallback::new(
                "indexes:nodes",
                "a node type has no interned name",
            ));
            continue;
        };
        out.push(owned(
            type_section_name(name),
            KIND_TYPE_MEMBERS,
            SectionDtype::U32,
            scan.members.len(),
            scan.members.iter().flat_map(|n| n.to_le_bytes()).collect(),
        ));
        if let (false, Some(reason)) = (scan.uniform, scan.abandoned) {
            fallbacks.push(MappedFallback::new(format!("indexes:id:{name}"), reason));
        }
        let Some(kind) = scan
            .kind
            .filter(|_| scan.uniform && !(scan.ids.is_empty() && scan.strings.entries.is_empty()))
        else {
            continue;
        };
        if kind == IdKind::Str {
            push_string_ids(name, &mut scan.strings, out);
            continue;
        }
        scan.ids.sort_unstable();
        let shadowed = keep_last_per_id(&mut scan.ids, |(id, _)| *id, |(_, node)| *node);
        let count = scan.ids.len();
        let (dtype, keys): (SectionDtype, Vec<u8>) = match kind {
            IdKind::Int64 => (
                SectionDtype::I64,
                scan.ids
                    .iter()
                    .flat_map(|(id, _)| id.to_le_bytes())
                    .collect(),
            ),
            IdKind::Unique => (
                SectionDtype::U32,
                scan.ids
                    .iter()
                    .flat_map(|(id, _)| (*id as u32).to_le_bytes())
                    .collect(),
            ),
            IdKind::Str => unreachable!("string ids are written by push_string_ids"),
        };
        out.push(owned(
            id_section_name(name, "keys"),
            KIND_ID_KEYS,
            dtype,
            count,
            keys,
        ));
        out.push(owned(
            id_section_name(name, "nodes"),
            KIND_ID_NODES,
            SectionDtype::U32,
            count,
            scan.ids.iter().flat_map(|(_, n)| n.to_le_bytes()).collect(),
        ));
        push_shadowed(name, shadowed, out);
    }
}

/// Of each run of entries with one id (the slice is sorted by id, then node),
/// keep the last: the node the lazy build keeps, which inserts a type's nodes
/// in ascending order and lets a later node replace an earlier one. The nodes
/// of the entries dropped, in the order met.
fn keep_last_per_id<T, K: PartialEq>(
    entries: &mut Vec<T>,
    key: impl Fn(&T) -> K,
    node: impl Fn(&T) -> u32,
) -> Vec<u32> {
    let mut shadowed = Vec::new();
    let mut kept = 0;
    for read in 0..entries.len() {
        if read + 1 == entries.len() || key(&entries[read]) != key(&entries[read + 1]) {
            entries.swap(kept, read);
            kept += 1;
        } else {
            shadowed.push(node(&entries[read]));
        }
    }
    entries.truncate(kept);
    shadowed
}

/// The section naming the nodes [`keep_last_per_id`] dropped, or nothing.
fn push_shadowed(name: &str, mut shadowed: Vec<u32>, out: &mut Vec<MappableSection<'static>>) {
    if shadowed.is_empty() {
        return;
    }
    shadowed.sort_unstable();
    out.push(owned(
        id_section_name(name, "shadowed"),
        KIND_ID_SHADOWED,
        SectionDtype::U32,
        shadowed.len(),
        shadowed.iter().flat_map(|n| n.to_le_bytes()).collect(),
    ));
}

/// The id sections of a type whose ids are all strings.
fn push_string_ids(name: &str, ids: &mut StringIds, out: &mut Vec<MappableSection<'static>>) {
    let mut entries = std::mem::take(&mut ids.entries);
    entries.sort_unstable_by(|a, b| ids.key(a).cmp(ids.key(b)).then(a.2.cmp(&b.2)));
    let shadowed = keep_last_per_id(&mut entries, |entry| ids.key(entry), |entry| entry.2);
    push_shadowed(name, shadowed, out);
    let count = entries.len();
    let (keys, nodes) = string_keys::encode(entries.iter().map(|e| (ids.key(e), e.2)));
    out.push(owned(
        id_section_name(name, "keys"),
        KIND_ID_STRINGS,
        SectionDtype::U8,
        keys.len(),
        keys,
    ));
    out.push(owned(
        id_section_name(name, "nodes"),
        KIND_ID_NODES,
        SectionDtype::U32,
        count,
        nodes,
    ));
}

/// Counts per node for one connection type: `(by_target, by_source)`.
pub(super) type PeerCounts = (Vec<(u32, u32)>, Vec<(u32, u32)>);

/// Bytes of dense per-node counters a save may allocate while counting peers;
/// past it the counting falls back to hash maps.
const DENSE_BUDGET: usize = 512 * 1024 * 1024;

/// One pass over the edges counting, per connection type, how many land on and
/// leave each node. Dense `u32` counters while they fit [`DENSE_BUDGET`]: a
/// random increment into an array is several times cheaper than a hash insert,
/// and reading the array back yields the pairs already sorted.
pub(super) fn count_peers_dense(graph: &DirGraph) -> Option<FxHashMap<InternedKey, PeerCounts>> {
    let bound = graph.graph.node_bound();
    let mut dense: FxHashMap<InternedKey, (Vec<u32>, Vec<u32>)> = FxHashMap::default();
    for (source, target, conn) in graph.graph.edge_endpoint_keys() {
        if !dense.contains_key(&conn) {
            if (dense.len() + 1).saturating_mul(bound).saturating_mul(8) > DENSE_BUDGET {
                return None;
            }
            dense.insert(conn, (vec![0; bound], vec![0; bound]));
        }
        let (by_target, by_source) = dense.get_mut(&conn)?;
        // An endpoint past the node array is a crafted file; a save refuses it
        // before this runs, and the count skips it rather than index out of bounds.
        if let (Some(t), Some(s)) = (
            by_target.get_mut(target.index()),
            by_source.get_mut(source.index()),
        ) {
            *t += 1;
            *s += 1;
        }
    }
    let pairs = |counts: Vec<u32>| -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        out.extend(
            counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count != 0)
                .map(|(node, count)| (node as u32, *count)),
        );
        out
    };
    Some(
        dense
            .into_iter()
            .map(|(conn, (by_target, by_source))| (conn, (pairs(by_target), pairs(by_source))))
            .collect(),
    )
}

/// [`count_peers_dense`] with hash maps, for a graph whose node array is too
/// wide for dense counters across its connection types.
pub(super) fn count_peers_hashed(graph: &DirGraph) -> FxHashMap<InternedKey, PeerCounts> {
    type Counts = FxHashMap<u32, u32>;
    let mut per_type: FxHashMap<InternedKey, (Counts, Counts)> = FxHashMap::default();
    for (source, target, conn) in graph.graph.edge_endpoint_keys() {
        let (by_target, by_source) = per_type.entry(conn).or_default();
        *by_target.entry(target.index() as u32).or_insert(0) += 1;
        *by_source.entry(source.index() as u32).or_insert(0) += 1;
    }
    let sorted = |counts: Counts| -> Vec<(u32, u32)> {
        let mut pairs: Vec<(u32, u32)> = counts.into_iter().collect();
        pairs.sort_unstable_by_key(|(node, _)| *node);
        pairs
    };
    per_type
        .into_iter()
        .map(|(conn, (by_target, by_source))| (conn, (sorted(by_target), sorted(by_source))))
        .collect()
}

fn push_peer_sections(
    graph: &DirGraph,
    out: &mut Vec<MappableSection<'static>>,
    fallbacks: &mut Vec<MappedFallback>,
) {
    let per_type = {
        let _guard = graph.graph.begin_query();
        count_peers_dense(graph).unwrap_or_else(|| count_peers_hashed(graph))
    };
    if per_type.len() > MAX_TYPES {
        fallbacks.push(MappedFallback::new(
            "indexes:peers",
            "more than 1,024 connection types: no peer counts are recorded",
        ));
        return;
    }
    for (conn, (by_target, by_source)) in per_type {
        let Some(name) = graph.interner.try_resolve(conn) else {
            continue;
        };
        for (part, pairs) in [("out", by_target), ("in", by_source)] {
            let mut bytes = Vec::with_capacity(pairs.len() * 8);
            for (node, count) in &pairs {
                bytes.extend_from_slice(&node.to_le_bytes());
                bytes.extend_from_slice(&count.to_le_bytes());
            }
            out.push(owned(
                peer_section_name(name, part),
                KIND_PEERS,
                SectionDtype::U32,
                pairs.len() * 2,
                bytes,
            ));
        }
    }
}

/// Where a load puts one index section.
///
/// Into memory storage (or under `topology="memory"`) nothing reads them, so a
/// file-backed section is left unread in the mapping and a byte-buffer load
/// keeps only the validated range. Into mapped storage they follow
/// `LoadOptions::topology`, except that Windows always copies: an open mapping
/// there blocks the rename a save performs.
pub(super) fn place(
    entry: &MapEntry,
    file: Option<&Arc<MappedFile>>,
    policy: MapPolicy,
    into_mapped: bool,
) -> Placement {
    if !into_mapped {
        return if file.is_some() {
            Placement::Mapped
        } else {
            Placement::Deferred
        };
    }
    let ctx = MapContext::host(file.map(|f| &**f), entry.len);
    if !cfg!(windows) && resolve_mapping(policy, 0, &ctx) {
        Placement::Mapped
    } else {
        Placement::Heap
    }
}

/// The index sections of an opened table that a mapped load can serve.
#[derive(Default)]
pub(super) struct IndexSections {
    types: Vec<(String, MappedSection)>,
    /// `(type, keys, nodes, nodes that share an id with a later node)`.
    ids: Vec<(String, MappedSection, MappedSection, u64)>,
    peers: Vec<(String, MappedSection, MappedSection)>,
}

impl IndexSections {
    /// Collect the index sections a table holds. A section of another kind or
    /// element type than its name promises, or one missing its partner, is
    /// ignored: an index is derived data, and the loader rebuilds whatever the
    /// file does not serve.
    pub(super) fn from_table(table: &MappableTable) -> Self {
        let partner = |name: String, kind: u8, dtype: SectionDtype| {
            table
                .get(&name)
                .filter(|s| s.owner == OWNER && s.kind == kind && s.dtype == dtype)
                .cloned()
        };
        let mut found = IndexSections::default();
        for section in table.iter().filter(|s| s.owner == OWNER) {
            let name = section.name.as_str();
            match (section.kind, section.dtype) {
                (KIND_TYPE_MEMBERS, SectionDtype::U32) => {
                    if let Some(stem) = name.strip_prefix("idx:type:") {
                        found.types.push((unescape(stem), section.clone()));
                    }
                }
                (KIND_ID_KEYS, SectionDtype::I64 | SectionDtype::U32)
                | (KIND_ID_STRINGS, SectionDtype::U8) => {
                    let stem = name
                        .strip_prefix("idx:id:")
                        .and_then(|rest| rest.strip_suffix(":keys"));
                    if let Some(stem) = stem {
                        let nodes = partner(
                            format!("idx:id:{stem}:nodes"),
                            KIND_ID_NODES,
                            SectionDtype::U32,
                        );
                        let shadowed = partner(
                            format!("idx:id:{stem}:shadowed"),
                            KIND_ID_SHADOWED,
                            SectionDtype::U32,
                        )
                        .map_or(0, |section| section.elem_count);
                        if let Some(nodes) = nodes {
                            found
                                .ids
                                .push((unescape(stem), section.clone(), nodes, shadowed));
                        }
                    }
                }
                (KIND_PEERS, SectionDtype::U32) => {
                    let stem = name
                        .strip_prefix("idx:peer:")
                        .and_then(|rest| rest.strip_suffix(":out"));
                    if let Some(stem) = stem {
                        let inn =
                            partner(format!("idx:peer:{stem}:in"), KIND_PEERS, SectionDtype::U32);
                        if let Some(inn) = inn {
                            found.peers.push((unescape(stem), section.clone(), inn));
                        }
                    }
                }
                _ => {}
            }
        }
        found
    }

    pub(super) fn has_types(&self) -> bool {
        !self.types.is_empty()
    }

    /// Install the type-member and id indexes on `graph`, whose topology was
    /// just decoded and whose type indexes are not yet built. Returns `false`
    /// without touching `graph` when the sections do not describe it (a type
    /// count that disagrees with the topology); the caller then rebuilds.
    pub(super) fn install_node_indexes(&mut self, graph: &mut DirGraph) -> io::Result<bool> {
        let bound = graph.graph.node_bound();
        let total: usize = self
            .types
            .iter()
            .map(|(_, section)| section.as_bytes().len() / 4)
            .sum();
        if total != graph.graph.node_count() {
            return Ok(false);
        }
        let types = TypeIndexBase::from_sections(std::mem::take(&mut self.types), bound)?;
        let ids: Vec<_> = std::mem::take(&mut self.ids)
            .into_iter()
            .filter(|(name, _, nodes, shadowed)| {
                types.slice_for(name).is_some_and(|members| {
                    members.len() / 4 == (nodes.elem_count + shadowed) as usize
                })
            })
            .map(|(name, keys, nodes, _)| (name, keys, nodes))
            .collect();
        let id_base = IdIndexBase::from_sections(ids, bound)?;
        graph.type_indices = TypeIndexStore::from_base(types);
        graph.id_indices = IdIndexStore::from_base(id_base);
        Ok(true)
    }

    /// The peer-count base for a mapped backend: counts for every connection
    /// type the file recorded, keyed the way the backend keys its caches.
    pub(super) fn peer_base(&self, graph: &DirGraph) -> io::Result<PeerBase> {
        let triples = self
            .peers
            .iter()
            .map(|(name, out, inn)| {
                (
                    InternedKey::from_str(name).as_u64(),
                    out.clone(),
                    inn.clone(),
                )
            })
            .collect();
        PeerBase::from_sections(triples, graph.graph.node_bound())
    }
}

#[cfg(test)]
mod key_class_tests {
    use super::*;

    fn scan() -> TypeScan {
        TypeScan {
            uniform: true,
            ..TypeScan::default()
        }
    }

    #[test]
    fn unique_and_int64_ids_are_one_key_class_and_a_string_is_not() {
        let mut mixed = scan();
        mixed.see(&Value::UniqueId(7), 0);
        mixed.see(&Value::Int64(1 << 40), 1);
        mixed.see(&Value::UniqueId(9), 2);
        assert!(mixed.uniform && mixed.kind == Some(IdKind::Int64));
        assert_eq!(mixed.ids, vec![(7, 0), (1 << 40, 1), (9, 2)]);

        let mut unique = scan();
        unique.see(&Value::UniqueId(7), 0);
        assert_eq!(unique.kind, Some(IdKind::Unique));

        let mut text = scan();
        text.see(&Value::UniqueId(7), 0);
        text.see(&Value::String("7".into()), 1);
        assert!(!text.uniform && text.ids.is_empty());
    }

    #[test]
    fn a_repeated_id_keeps_its_last_node_and_names_the_others() {
        let mut ids = vec![(1i64, 4u32), (1, 9), (2, 3), (3, 1), (3, 2), (3, 8)];
        let shadowed = keep_last_per_id(&mut ids, |(id, _)| *id, |(_, node)| *node);
        assert_eq!(ids, vec![(1, 9), (2, 3), (3, 8)]);
        assert_eq!(shadowed, vec![4, 1, 2]);
    }
}
