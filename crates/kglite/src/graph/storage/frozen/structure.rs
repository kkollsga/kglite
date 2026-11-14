//! The structural checks the eager verify tier adds to a frozen topology: the
//! adjacency arrays against the endpoint records and against each other.
//! A lazy open runs none of them (each is a pass over pages it never reads),
//! so a file that fails one still answers, possibly wrongly.

use petgraph::Direction;

use super::{FrozenGraph, VACANT};

/// A bitset over edge slots that reports a slot seen twice.
struct SlotSet(Vec<u64>);

impl SlotSet {
    fn new(slots: usize) -> Self {
        SlotSet(vec![0; slots.div_ceil(64)])
    }

    /// `false` when `slot` was already in the set.
    fn insert(&mut self, slot: usize) -> bool {
        let (word, bit) = (slot >> 6, 1u64 << (slot & 63));
        let fresh = self.0[word] & bit == 0;
        self.0[word] |= bit;
        fresh
    }

    fn len(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }
}

impl FrozenGraph {
    /// Every check of this module, first failure first: the endpoint array,
    /// then each direction's adjacency and the type mask.
    pub(crate) fn check_structure(&self) -> Result<(), String> {
        self.validated_endpoints().map_err(|e| e.to_string())?;
        self.check_direction(Direction::Outgoing)?;
        self.check_direction(Direction::Incoming)
    }

    /// One direction's adjacency: each entry names a live edge slot whose
    /// endpoints and type ordinal match, each live edge appears exactly once,
    /// each node's run is ascending by ordinal (the typed-run lookup binary
    /// searches it), and the node's type mask covers every ordinal in its run.
    /// Needs [`Self::validated_endpoints`] to have passed.
    fn check_direction(&self, dir: Direction) -> Result<(), String> {
        let (name_edges, name_ty) = match dir {
            Direction::Outgoing => ("topo:out_edges", "topo:out_ty"),
            Direction::Incoming => ("topo:in_edges", "topo:in_ty"),
        };
        let (_, edges, tys) = self.dir_arrays(dir);
        let endpoints = self.endpoint_slots();
        let mut seen = SlotSet::new(endpoints.len());
        for node in 0..self.node_bound_len() {
            let (lo, hi) = self.run(node, dir);
            let mut previous = 0u16;
            let mut mask = 0u32;
            for i in lo..hi {
                let (entry, ty) = (edges[i], tys[i]);
                let live = endpoints
                    .get(entry.edge_idx as usize)
                    .filter(|ep| ep.src != VACANT);
                let Some(ep) = live else {
                    return Err(format!(
                        "{name_edges} entry {i} (node {node}): edge slot {} holds no edge",
                        entry.edge_idx
                    ));
                };
                let (own, peer) = match dir {
                    Direction::Outgoing => (ep.src, ep.dst),
                    Direction::Incoming => (ep.dst, ep.src),
                };
                if own as usize != node || peer != entry.peer || ep.ty != ty {
                    return Err(format!(
                        "{name_edges} entry {i} (node {node}): edge slot {} disagrees with the \
                         endpoint record on the node, peer or type ordinal",
                        entry.edge_idx
                    ));
                }
                if ty < previous {
                    return Err(format!(
                        "{name_ty} entry {i} (node {node}): the run is not ascending by type ordinal"
                    ));
                }
                previous = ty;
                mask |= 1 << (ty & 31);
                if !seen.insert(entry.edge_idx as usize) {
                    return Err(format!(
                        "{name_edges} entry {i} (node {node}): edge slot {} is listed twice",
                        entry.edge_idx
                    ));
                }
            }
            if let Some(masks) = &self.type_mask {
                let at = 2 * node + usize::from(dir == Direction::Incoming);
                if masks
                    .as_slice()
                    .get(at)
                    .is_none_or(|stored| mask & !stored != 0)
                {
                    return Err(format!(
                        "topo:type_mask entry {at} (node {node}): clear for an ordinal the \
                         node's run holds"
                    ));
                }
            }
        }
        if seen.len() != self.edge_count {
            return Err(format!(
                "{name_edges} lists {} edges but the file counts {}",
                seen.len(),
                self.edge_count
            ));
        }
        Ok(())
    }
}
