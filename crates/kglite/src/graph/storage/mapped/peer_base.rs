//! Per-edge-type peer counts persisted in a v9 `.kgl`, served to a mapped
//! graph without the O(edges) scan that otherwise builds them.
//!
//! For each connection type the file holds two sections of little-endian `u32`
//! pairs `(node, count)`, ascending by node: `by_target` counts the edges of
//! the type landing on each node (what an outgoing grouped aggregate groups
//! by), `by_source` the edges leaving each node. The keys of `by_source` are
//! therefore the sorted distinct sources of the type, so no separate source
//! list is stored.
//!
//! A base is valid only for the graph exactly as the file recorded it. Every
//! edge mutation drops it (`MappedGraph::invalidate_type_index`), after which
//! the counts are rebuilt lazily from the edges as before.

use std::collections::HashMap;
use std::io;

use petgraph::Direction;

use crate::graph::io::file::mappable::MappedSection;

fn invalid(message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid peer-count section: {message}"),
    )
}

/// The two count sections of one connection type.
#[derive(Debug, Clone)]
pub(crate) struct ConnPeers {
    by_target: MappedSection,
    by_source: MappedSection,
}

/// Pairs of `(node, count)` in a section's bytes.
fn pairs(bytes: &[u8]) -> impl Iterator<Item = (u32, u32)> + '_ {
    bytes.as_chunks::<8>().0.iter().map(|chunk| {
        (
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
        )
    })
}

/// Total edges a section counts, after checking its pairs ascend strictly
/// below `node_bound` with a positive count each.
fn checked_total(section: &MappedSection, node_bound: usize) -> io::Result<u64> {
    section.verify()?;
    let bytes = section.as_bytes();
    if !bytes.len().is_multiple_of(8) {
        return Err(invalid("a section is not a run of (node, count) pairs"));
    }
    let (mut previous, mut total) = (None, 0u64);
    for (node, count) in pairs(bytes) {
        if previous.is_some_and(|prior| node <= prior) || node as usize >= node_bound || count == 0
        {
            return Err(invalid(
                "peers are not strictly ascending, name a node outside the graph, or count zero",
            ));
        }
        previous = Some(node);
        total += u64::from(count);
    }
    section.release_resident();
    Ok(total)
}

impl ConnPeers {
    /// The counts grouped by the far end of `dir`: `Outgoing` groups by target,
    /// `Incoming` by source, the convention of `count_edges_grouped_by_peer`.
    pub(crate) fn counts(&self, dir: Direction) -> HashMap<u32, i64> {
        let section = match dir {
            Direction::Outgoing => &self.by_target,
            Direction::Incoming => &self.by_source,
        };
        let bytes = section.as_bytes();
        let mut map = HashMap::with_capacity(bytes.len() / 8);
        map.extend(pairs(bytes).map(|(node, count)| (node, i64::from(count))));
        map
    }

    /// The distinct source nodes of the type, ascending, at most `max` of them.
    pub(crate) fn sources(&self, max: Option<usize>) -> Vec<u32> {
        let limit = max.unwrap_or(usize::MAX);
        pairs(self.by_source.as_bytes())
            .take(limit)
            .map(|(node, _)| node)
            .collect()
    }
}

/// Every connection type's persisted counts, by `InternedKey::as_u64`.
#[derive(Debug, Default)]
pub(crate) struct PeerBase {
    by_conn: HashMap<u64, ConnPeers>,
}

impl PeerBase {
    /// A base over `(connection type key, by_target, by_source)` triples.
    /// Each section is CRC-checked and structurally validated here, and the
    /// two sections of a type must count the same number of edges, so a
    /// damaged file is refused at load instead of answering a query wrongly.
    pub(crate) fn from_sections(
        triples: Vec<(u64, MappedSection, MappedSection)>,
        node_bound: usize,
    ) -> io::Result<Self> {
        let mut by_conn = HashMap::with_capacity(triples.len());
        for (key, by_target, by_source) in triples {
            let targets = checked_total(&by_target, node_bound)?;
            let sources = checked_total(&by_source, node_bound)?;
            if targets != sources {
                return Err(invalid(
                    "the target and source sections of a type count different edge totals",
                ));
            }
            if by_conn
                .insert(
                    key,
                    ConnPeers {
                        by_target,
                        by_source,
                    },
                )
                .is_some()
            {
                return Err(invalid("a connection type appears twice"));
            }
        }
        Ok(PeerBase { by_conn })
    }

    pub(crate) fn get(&self, conn: u64) -> Option<&ConnPeers> {
        self.by_conn.get(&conn)
    }

    /// The connection types the base holds counts for, by `InternedKey::as_u64`.
    pub(crate) fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.by_conn.keys().copied()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_conn.is_empty()
    }
}
