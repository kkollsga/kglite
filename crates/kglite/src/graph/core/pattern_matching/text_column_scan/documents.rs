//! The driver's second reader: the documents a scope holds, as the bytes the
//! column stores, for callers that rank or count rather than match.
//!
//! It shares the scan's eligibility and row-to-node proof, so a column that
//! answers a match answers here, and a decline means the same thing: the
//! caller reads the values through `NodeView`, which gives the same documents.
//! A document is a node's current string value; a null, absent or relocated-away
//! value follows the same rules as the scan (relocated rows read their overlay
//! string, null rows are not documents). Lists are not string columns, so a
//! property holding them declines and arrives through the per-row route.

use petgraph::graph::NodeIndex;

use super::{
    check_cardinality, eligible_parts, scope_rows, walk_bucket, DirGraph, InternedKey, Interrupt,
    TextDecline, TextScanInterrupted,
};

/// Each document's node with its text bytes.
pub(crate) type Documents<'g> = Vec<(NodeIndex, &'g [u8])>;

/// The live nodes of `node_type` (those of `scope` when one is given; repeats
/// and other types' nodes ignored) whose property holds a string, in node
/// order, each with that string's UTF-8 bytes borrowed from the column (or from
/// the overlay of a relocated row).
///
/// Nodes whose value is null or not text are absent from the list.
pub(crate) fn documents<'g>(
    graph: &'g DirGraph,
    node_type: &str,
    property: &str,
    scope: Option<&[NodeIndex]>,
    interrupt: Interrupt,
) -> Result<Result<Documents<'g>, TextDecline>, TextScanInterrupted> {
    let type_key = InternedKey::from_str(node_type);
    let parts = match eligible_parts(graph, node_type, property, type_key) {
        Ok(parts) => parts,
        Err(decline) => return Ok(Err(decline)),
    };
    let total_rows = parts.last().map_or(0, |p| p.first_row + p.rows);
    let mut walk = match scope {
        Some(nodes) => match scope_rows(graph, nodes, type_key, total_rows) {
            Ok(pairs) => pairs,
            Err(decline) => return Ok(Err(decline)),
        },
        None => {
            if let Err(decline) = check_cardinality(graph, node_type, &parts) {
                return Ok(Err(decline));
            }
            match walk_bucket(graph, node_type, total_rows, &interrupt)? {
                Ok(walk) => walk,
                Err(decline) => return Ok(Err(decline)),
            }
        }
    };
    if scope.is_some() {
        walk.sort_unstable_by_key(|&(node, _)| node);
        walk.dedup_by_key(|&mut (node, _)| node);
    }
    let mut out = Vec::with_capacity(walk.len());
    for (polled, &(node, row)) in walk.iter().enumerate() {
        if polled % 64 == 0 && interrupt.exceeded() {
            return Err(TextScanInterrupted);
        }
        let at = parts.partition_point(|p| p.first_row + p.rows <= row);
        let Some(part) = parts.get(at) else {
            return Ok(Err(TextDecline::Cardinality));
        };
        let local = row - part.first_row;
        let Some(col) = part.col else { continue };
        if part.tombstoned(local) || col.nulls[local as usize] != 0 {
            continue;
        }
        match col.relocated.get(&local) {
            Some(text) => out.push((node, text.as_bytes())),
            None => {
                let (from, to) = (
                    col.offsets[local as usize] as usize,
                    col.offsets[local as usize + 1] as usize,
                );
                out.push((node, &col.data[from..to]));
            }
        }
    }
    Ok(Ok(out))
}
