//! Path-variable binding for `MATCH` and `OPTIONAL MATCH`: assembling each
//! `p = …` assignment's [`PathBinding`] from the bindings its pattern left on
//! a row.
//!
//! A row carries a pattern's hops in pieces: every tracked fixed hop on the
//! `__fixed_path` trail (in pattern order), and each variable-length segment
//! under its own binding — the relationship variable, or
//! `__anon_vlpath_{i}` for an unnamed one (`i` being the element index of the
//! node that follows the segment). A path is those pieces stitched back into
//! pattern order. The pieces are read by name, never by position: a row also
//! holds the paths and segments of earlier clauses, and "the first path
//! binding" was one of those whenever an earlier clause had bound one.

use super::*;
use crate::datatypes::values::RelationshipIncarnation;
use crate::graph::core::pattern_matching::{EdgePattern, Pattern, PatternElement};

/// Row key of the fixed-hop trail the matcher records for a pattern's
/// non-variable-length hops (see `match_clause::pattern_match_to_row`).
const FIXED_TRAIL: &str = "__fixed_path";

impl<'a> CypherExecutor<'a> {
    /// Whether `pred` may read one of `clause`'s path variables, so the
    /// variables must be bound before it runs.
    ///
    /// [`match_clause::predicate_may_read_var`] over-approximates, which is
    /// the safe direction here: a spurious `true` costs binding paths on rows
    /// the predicate then rejects, never an answer.
    fn predicate_may_read_clause_path(clause: &MatchClause, pred: &Predicate) -> bool {
        clause.path_assignments.iter().any(|pa| {
            !pa.is_shortest_path && match_clause::predicate_may_read_var(pred, &pa.variable)
        })
    }

    /// Whether the leading MATCH's row loop binds `clause`'s path variables
    /// per row, before `inline_where` runs, instead of in the post-pass.
    /// Decided once per clause: rows a path-free predicate rejects never pay
    /// for a path.
    pub(super) fn binds_paths_before_where(
        clause: &MatchClause,
        inline_where: Option<&Predicate>,
    ) -> bool {
        clause.patterns.len() == 1
            && inline_where.is_some_and(|pred| Self::predicate_may_read_clause_path(clause, pred))
    }

    /// Bind every non-shortestPath path variable of `clause` on `row`.
    /// shortestPath assignments are bound by their own executor route.
    pub(super) fn bind_row_paths(&self, clause: &MatchClause, row: &mut ResultRow) {
        for pa in &clause.path_assignments {
            if pa.is_shortest_path {
                continue;
            }
            let Some(pattern) = clause.patterns.get(pa.pattern_index) else {
                continue;
            };
            if let Some(path) = self.assemble_pattern_path(pattern, row) {
                row.path_bindings.insert(pa.variable.clone(), path);
            }
        }
    }

    /// Record every path variable of `clause` as NULL on a row the clause
    /// null-extended, unless the row already binds the name.
    pub(super) fn null_pad_path_vars(clause: &MatchClause, row: &mut ResultRow) {
        for pa in &clause.path_assignments {
            if !row.path_bindings.contains_key(&pa.variable)
                && !row.projected.contains_key(&pa.variable)
            {
                row.projected.insert(pa.variable.clone(), Value::Null);
            }
        }
    }

    /// The path `pattern` matched on `row`, or `None` when a piece is missing.
    pub(super) fn assemble_pattern_path(
        &self,
        pattern: &Pattern,
        row: &ResultRow,
    ) -> Option<PathBinding> {
        let (mut fixed_edges, mut var_length_edges) = (0usize, 0usize);
        for element in &pattern.elements {
            if let PatternElement::Edge(ep) = element {
                if ep.var_length.is_some() {
                    var_length_edges += 1;
                } else {
                    fixed_edges += 1;
                }
            }
        }
        // The trail belongs to this pattern only if it covers exactly this
        // pattern's fixed hops; otherwise it is an earlier clause's.
        let fixed_trail = row
            .path_bindings
            .get(FIXED_TRAIL)
            .filter(|trail| fixed_edges > 0 && trail.hops == fixed_edges);
        if var_length_edges == 0 {
            // No trail (e.g. a zero-length path): rebuild from the bindings.
            return fixed_trail
                .cloned()
                .or_else(|| self.synthesize_path_from_pattern(pattern, row));
        }
        if fixed_edges == 0 && var_length_edges == 1 {
            let (index, ep) = pattern
                .elements
                .iter()
                .enumerate()
                .find_map(|(i, e)| match e {
                    PatternElement::Edge(ep) => Some((i, ep)),
                    PatternElement::Node(_) => None,
                })?;
            // The segment reads as a relationship list; the path over it does not.
            return var_length_segment(ep, index, row).map(|segment| PathBinding {
                relationship_list: false,
                ..segment.clone()
            });
        }
        stitch_path(pattern, fixed_trail?, row)
    }
}

/// The binding a variable-length segment left on `row`.
fn var_length_segment<'r>(
    ep: &EdgePattern,
    element_index: usize,
    row: &'r ResultRow,
) -> Option<&'r PathBinding> {
    match ep.variable.as_deref() {
        Some(name) => row.path_bindings.get(name),
        None => row
            .path_bindings
            .get(&format!("__anon_vlpath_{}", element_index + 1)),
    }
}

/// A pattern mixing fixed hops and variable-length segments: take the fixed
/// hops off `fixed_trail` in order and splice each segment in where it sits.
fn stitch_path(
    pattern: &Pattern,
    fixed_trail: &PathBinding,
    row: &ResultRow,
) -> Option<PathBinding> {
    let mut source = None;
    let mut path = Vec::new();
    let mut tokens: Vec<Option<RelationshipIncarnation>> = Vec::new();
    let mut tracked = fixed_trail.hop_incarnations.is_some();
    let mut fixed_pos = 0usize;
    for (index, element) in pattern.elements.iter().enumerate() {
        let PatternElement::Edge(ep) = element else {
            continue;
        };
        if ep.var_length.is_some() {
            let segment = var_length_segment(ep, index, row)?;
            source.get_or_insert(segment.source);
            tracked |= segment.hop_incarnations.is_some();
            for hop_index in 0..segment.path.len() {
                tokens.push(segment.hop_incarnation(hop_index));
            }
            path.extend_from_slice(&segment.path);
        } else {
            source.get_or_insert(fixed_trail.source);
            path.push(*fixed_trail.path.get(fixed_pos)?);
            tokens.push(fixed_trail.hop_incarnation(fixed_pos));
            fixed_pos += 1;
        }
    }
    Some(PathBinding {
        relationship_list: false,
        source: source?,
        hops: path.len(),
        path,
        hop_incarnations: tracked.then_some(tokens),
    })
}
