//! Load-time structure check of a disk graph's mapped arrays.
//!
//! The arrays are mapped, not parsed, so a damaged file loads fine and faults
//! on the first query that reaches the bad entry (an interner `resolve` panic,
//! a `MmapOrVec::get` bound assert). The check reads each array once and
//! rejects, at load, every value those readers would trust blindly: type ids
//! the interner does not know, endpoints past the node range, and CSR offsets
//! that are not an ascending prefix of the edge array. Edge contents (peer and
//! edge index of each CSR entry) are not walked; that would be an O(E) random
//! pass and no reader panics on them.

use super::csr::{CsrEdge, EdgeEndpoints, TOMBSTONE_EDGE};
use super::graph::DiskGraph;
use crate::graph::schema::{InternedKey, StringInterner};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;

impl DiskGraph {
    /// Run every structure check; the first failure names its file.
    /// Skips the CSR check while a deferred build owns the edges.
    pub(crate) fn check_structure(&self, interner: &StringInterner) -> Result<(), String> {
        self.check_node_slots(interner)?;
        self.check_edge_endpoints(interner)?;
        self.edge_properties
            .check_base_offsets()
            .map_err(|e| e.to_string())?;
        if !self.defer_csr {
            check_offsets("out_offsets", &self.out_offsets, &self.out_edges)?;
            check_offsets("in_offsets", &self.in_offsets, &self.in_edges)?;
        }
        Ok(())
    }

    /// Every alive slot carries a node type the interner resolves.
    /// Free slots keep stale data and are not read.
    fn check_node_slots(&self, interner: &StringInterner) -> Result<(), String> {
        let mut known = None;
        for (i, slot) in self.node_slots.as_slice().iter().enumerate() {
            if !slot.is_alive() || known == Some(slot.node_type) {
                continue;
            }
            if interner
                .try_resolve(InternedKey::from_u64(slot.node_type))
                .is_none()
            {
                return Err(format!(
                    "node_slots: slot {i} has unknown node type id {}",
                    slot.node_type
                ));
            }
            known = Some(slot.node_type);
        }
        Ok(())
    }

    /// Every live endpoint record names a known connection type and nodes
    /// inside the slot range, and the edge counter does not exceed the array.
    fn check_edge_endpoints(&self, interner: &StringInterner) -> Result<(), String> {
        let endpoints: &[EdgeEndpoints] = self.edge_endpoints.as_slice();
        if self.next_edge_idx as usize > endpoints.len() + self.appended_edge_endpoints.len() {
            return Err(format!(
                "edge_endpoints: next_edge_idx {} exceeds the {} stored records",
                self.next_edge_idx,
                endpoints.len()
            ));
        }
        let nodes = self.node_slot_len() as u64;
        let mut known = None;
        for (i, ep) in endpoints.iter().enumerate() {
            if ep.source == TOMBSTONE_EDGE {
                continue;
            }
            if u64::from(ep.source) >= nodes || u64::from(ep.target) >= nodes {
                return Err(format!(
                    "edge_endpoints: edge {i} joins nodes {} and {} but only {nodes} node slots exist",
                    ep.source, ep.target
                ));
            }
            if known != Some(ep.connection_type) {
                if interner
                    .try_resolve(InternedKey::from_u64(ep.connection_type))
                    .is_none()
                {
                    return Err(format!(
                        "edge_endpoints: edge {i} has unknown connection type id {}",
                        ep.connection_type
                    ));
                }
                known = Some(ep.connection_type);
            }
        }
        Ok(())
    }
}

/// An offsets array starts at 0, never decreases and ends inside the edge
/// array. It may be empty or shorter than the node count: overflow-only
/// nodes have no CSR range.
fn check_offsets(
    name: &str,
    offsets: &MmapOrVec<u64>,
    edges: &MmapOrVec<CsrEdge>,
) -> Result<(), String> {
    let offsets = offsets.as_slice();
    let Some(&first) = offsets.first() else {
        return Ok(());
    };
    if first != 0 {
        return Err(format!("{name}: first offset is {first}, expected 0"));
    }
    let mut prev = 0u64;
    for (i, &off) in offsets.iter().enumerate() {
        if off < prev {
            return Err(format!(
                "{name}: offset {i} ({off}) is below the previous offset ({prev})"
            ));
        }
        prev = off;
    }
    if prev > edges.len() as u64 {
        return Err(format!(
            "{name}: last offset {prev} is past the {} stored edges",
            edges.len()
        ));
    }
    Ok(())
}
