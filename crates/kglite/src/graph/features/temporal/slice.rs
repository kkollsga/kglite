//! The valid slice: a materialised graph holding only the elements valid at
//! one instant, for a consumer that needs a whole graph rather than a guarded
//! walk over the base (an algorithm that iterates `graph.graph` directly).
//!
//! A node is kept when the resolved [`ElementFilter`] admits it — valid under
//! every declared label it carries. A relationship is kept when the filter
//! admits it (its own declaration, keyed on its source node's type) **and**
//! both its endpoints are kept, the same hop rule the guarded matcher applies.
//! Every test goes through the filter's admit functions, which read a mask bit
//! where the endpoint index gave one and evaluate the bound properties where
//! it did not (Disk mode, an unreadable bound, the byte cap), so a slice is
//! correct whichever targets are indexed.
//!
//! The slice keeps user ids (`id(n)` answers as on the base) and renumbers
//! physical indexes: `to_base` takes a slice node back to its base node, so
//! rows an algorithm produces on the slice can be joined to the base. It
//! copies secondary labels, the ontology and the schema; it does **not** copy
//! temporal declarations (the slice is already as of its instant),
//! embeddings, text or vector indexes.
//!
//! The copy is a fresh in-memory graph with its own column stores, holding
//! only the kept nodes' rows; it never shares the base's stores, which in
//! mapped and Disk mode are file-backed and would be copied onto the heap
//! whole by the first write.
//!
//! Two caps bound it. Every mode: the slice's bytes (see
//! [`ValidSlice::bytes`]) against [`SLICE_BYTE_CAP`] — over it the slice is
//! refused, and the cache of slices evicts oldest first to stay under it.
//! Disk mode also counts admitted elements against
//! [`DISK_SLICE_ELEMENT_CAP`] while it walks, and refuses as soon as the walk
//! passes it. The walk holds only the admitted nodes and relationships —
//! nothing sized by the base graph — so on Disk its lists never grow past
//! the cap.

use std::sync::Arc;

use petgraph::graph::{EdgeIndex, NodeIndex};

use super::endpoint_index::SegmentKey;
use super::eval::Instant;
use crate::graph::core::graph_filter::ElementFilter;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::subgraph::copy_induced_subgraph;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::node_view::NodeView;
use crate::graph::storage::{GraphRead, StrField};

/// The bytes the cached slices of one graph may hold together, and the most
/// one slice may estimate at. [`SLICE_BYTE_CAP_ENV`] overrides it.
pub const SLICE_BYTE_CAP: usize = 128 << 20;

/// Environment variable that replaces [`SLICE_BYTE_CAP`] (a byte count),
/// read at each build.
pub(crate) const SLICE_BYTE_CAP_ENV: &str = "KGLITE_TEMPORAL_SLICE_MAX_BYTES";

/// The most admitted nodes plus relationships a Disk-mode slice may hold: a
/// slice is a heap copy, and Disk mode's heap must not grow with the graph.
/// It matches the procedures' full-graph limit (2M elements), since the
/// slice's consumers are those procedures. [`DISK_SLICE_CAP_ENV`] overrides
/// it.
pub const DISK_SLICE_ELEMENT_CAP: usize = 2_000_000;

/// Environment variable that replaces [`DISK_SLICE_ELEMENT_CAP`], read at
/// each build.
pub(crate) const DISK_SLICE_CAP_ENV: &str = "KGLITE_TEMPORAL_DISK_SLICE_MAX_ELEMENTS";

/// What one slice is cached under: the segment of every indexed target (equal
/// segments, equal valid sets), and — when some target is left to property
/// guards, whose valid set only the instant fixes — the instant itself.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SliceKey {
    pub(crate) segments: SegmentKey,
    pub(crate) instant: Option<Instant>,
}

/// A materialised graph of the elements valid at one instant; see the module
/// docs.
pub struct ValidSlice {
    graph: Arc<DirGraph>,
    /// Slice node index → base node index, ascending.
    to_base: Vec<NodeIndex>,
    bytes: usize,
}

impl std::fmt::Debug for ValidSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidSlice")
            .field("nodes", &self.to_base.len())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl ValidSlice {
    /// The slice itself, an ordinary in-memory graph with no declarations.
    pub fn graph(&self) -> &Arc<DirGraph> {
        &self.graph
    }

    /// The base node a slice node was copied from.
    pub fn to_base(&self, slice: NodeIndex) -> Option<NodeIndex> {
        self.to_base.get(slice.index()).copied()
    }

    /// The slice node copied from base node `base`, if it was kept.
    pub fn from_base(&self, base: NodeIndex) -> Option<NodeIndex> {
        self.to_base.binary_search(&base).ok().map(NodeIndex::new)
    }

    /// The slice's heap bytes, the figure the caps judge: an estimate of the
    /// node and relationship records with their property slots, plus the
    /// measured heap of the slice's own column stores (which hold only the
    /// kept nodes' rows). Heap payloads behind a relationship property value
    /// (a long string, a list) are not counted.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Per node: its record, its `to_base` entry and petgraph's two
/// adjacency heads.
const NODE_BYTES: usize = size_of::<NodeData>() + size_of::<NodeIndex>() + 2 * size_of::<u32>();
/// Per relationship: its record, its endpoints and petgraph's two next links.
const EDGE_BYTES: usize = size_of::<EdgeData>() + 4 * size_of::<u32>();
/// Per property slot.
const PROPERTY_BYTES: usize = size_of::<(InternedKey, crate::datatypes::values::Value)>();

/// The caps one build runs under.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SliceCaps {
    pub(crate) bytes: usize,
    /// Admitted nodes + relationships; `None` outside Disk mode.
    pub(crate) disk_elements: Option<usize>,
}

impl SliceCaps {
    /// The caps for `graph`, from the defaults or their environment
    /// overrides.
    pub(crate) fn for_graph(graph: &DirGraph) -> Self {
        let env = |name: &str| std::env::var(name).ok()?.parse::<usize>().ok();
        SliceCaps {
            bytes: env(SLICE_BYTE_CAP_ENV).unwrap_or(SLICE_BYTE_CAP),
            disk_elements: graph
                .graph
                .is_disk()
                .then(|| env(DISK_SLICE_CAP_ENV).unwrap_or(DISK_SLICE_ELEMENT_CAP)),
        }
    }
}

/// Counts a Disk-mode walk against its element cap.
struct ElementBudget {
    cap: Option<usize>,
    admitted: usize,
}

impl ElementBudget {
    fn admit(&mut self) -> Result<(), String> {
        self.admitted += 1;
        match self.cap {
            Some(cap) if self.admitted > cap => Err(format!(
                "the valid slice would hold more than {cap} nodes and relationships, the \
                 Disk-mode slice cap ({DISK_SLICE_CAP_ENV}); run it on an in-memory graph, or \
                 narrow the declared types"
            )),
            _ => Ok(()),
        }
    }
}

/// Build the valid slice of `base` under `filter` (`None`: the filter removes
/// nothing, so every element is kept), refused over `caps`. An unreadable
/// bound the filter met is an error naming the element.
pub(crate) fn slice_at(
    base: &DirGraph,
    filter: Option<&ElementFilter>,
    caps: SliceCaps,
) -> Result<ValidSlice, String> {
    let mut budget = ElementBudget {
        cap: caps.disk_elements,
        admitted: 0,
    };
    let mut nodes = Vec::new();
    let mut properties = 0usize;
    let mut payloads = 0usize;
    for idx in base.graph.node_indices() {
        if filter.is_some_and(|f| !f.admits_node(base, idx)) {
            continue;
        }
        budget.admit()?;
        nodes.push(idx);
        if let Some(view) = base.graph.node_view(idx) {
            properties += view.property_count();
            payloads += string_payload(&view);
        }
    }
    nodes.sort_unstable();
    let mut kept_edges: Vec<EdgeIndex> = Vec::new();
    for &source in &nodes {
        for edge in base.graph.edges(source) {
            // `weight()` only for kept edges: on Disk it parks a copy of the
            // relationship's properties until the guard drops.
            if nodes.binary_search(&edge.target()).is_err()
                || filter.is_some_and(|f| {
                    !f.admits_edge(base, edge.id(), edge.connection_type(), source)
                })
            {
                continue;
            }
            budget.admit()?;
            kept_edges.push(edge.id());
            properties += edge.weight().properties.len();
        }
    }
    if let Some(err) = filter.and_then(ElementFilter::error) {
        return Err(err.to_string());
    }
    // Refused before the copy on the records and the kept nodes' string
    // payloads, so a slice that cannot fit never builds the copy; the copy's
    // own stores are then measured and judged again.
    let records =
        nodes.len() * NODE_BYTES + kept_edges.len() * EDGE_BYTES + properties * PROPERTY_BYTES;
    over_cap(records + payloads, caps)?;
    kept_edges.sort_unstable();
    let (graph, _) =
        copy_induced_subgraph(base, &nodes, |edge| kept_edges.binary_search(&edge).is_ok())?;
    debug_assert!(
        graph
            .graph
            .node_indices()
            .enumerate()
            .all(|(i, idx)| idx.index() == i),
        "a fresh graph numbers the copied nodes densely, in copy order"
    );
    let stores: usize = graph
        .graph
        .column_stores_iter()
        .map(|(_, store)| store.heap_bytes())
        .sum();
    let bytes = records + stores;
    over_cap(bytes, caps)?;
    Ok(ValidSlice {
        graph: Arc::new(graph),
        to_base: nodes,
        bytes,
    })
}

/// The string bytes a copy of the node behind `view` puts in its store: its
/// id, title and string properties, measured one field at a time.
fn string_payload(view: &NodeView<'_>) -> usize {
    let len = |field: StrField<'_>| match field {
        StrField::Str(s) => s.len(),
        StrField::NotString | StrField::Absent => 0,
    };
    len(view.id_field())
        + len(view.title_field())
        + view
            .property_key_set()
            .into_iter()
            .map(|key| len(view.str_field(key)))
            .sum::<usize>()
}

fn over_cap(bytes: usize, caps: SliceCaps) -> Result<(), String> {
    if bytes > caps.bytes {
        return Err(format!(
            "the valid slice would take about {} MiB, over the {} MiB slice cap \
             ({SLICE_BYTE_CAP_ENV})",
            bytes.div_ceil(1 << 20),
            caps.bytes >> 20
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "slice_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "slice_memory_tests.rs"]
mod memory_tests;
