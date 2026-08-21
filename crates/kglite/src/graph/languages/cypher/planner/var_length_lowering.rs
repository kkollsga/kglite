//! Lower a fixed-length variable-hop segment (`*k..k`) to `k` explicit hops.
//!
//! `-[:R*2..2]->` and `-[:R]->()-[:R]->` ask the same question, but only the
//! second spelling reaches the fixed-pattern machinery: disjoint-trail
//! marking, `skip_target_type_check`, relationship-predicate pushdown,
//! start-node selection and every fusion pass bail the moment they see
//! `var_length.is_some()`. Rewriting the first spelling into the second at
//! plan time hands the whole fixed-pattern toolbox to a query the user
//! happened to write with a star.
//!
//! **Soundness.** Cypher paths are trails: nodes may repeat, relationships
//! may not. The fixed-hop matcher enforces that across hops already —
//! `matcher::reuses_bound_relationship` rejects a candidate whose edge is
//! already in the match's `exact_path`, which `extend_fixed_trail` fills for
//! every hop whose `needs_path_info` is set. The lowered hops therefore
//! reject exactly the walks the variable-length expansion rejects, including
//! the undirected "walk straight back over the same relationship" case.
//! [`super::annotations::pass_mark_disjoint_fixed_trails`] is the one pass
//! that turns that bookkeeping *off*, and it does so only when the hops'
//! relationship-type sets are pairwise disjoint — which `k >= 2` copies of
//! one edge pattern can never be, so a lowered multi-hop segment keeps its
//! trail. (A lowered `k == 1` may be marked, and correctly: a single
//! relationship cannot collide with itself.)

use super::super::ast::*;
use crate::graph::core::pattern_matching::{EdgePattern, NodePattern, Pattern, PatternElement};

/// Largest number of relationship elements a lowered pattern may contain.
///
/// `*8..8` becomes an eight-hop pattern, which the join-order and expansion
/// machinery handles the same way it handles a hand-written eight-hop
/// pattern. Beyond that the expanded IR stops paying for itself: the pattern
/// is quadratic to reorder, every pass walks it, and a segment that deep is
/// far more likely to be an unbounded exploration written with a big `k` than
/// a shape anyone wants materialised hop by hop. Above the ceiling the
/// segment stays variable-length — slower, never wrong.
const MAX_LOWERED_PATTERN_HOPS: usize = 8;

/// Rewrite every eligible `*k..k` segment in the query's MATCH patterns.
pub(super) fn lower_fixed_var_length_hops(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };
        // A path assignment (`p = ...`, `shortestPath(...)`) consumes the
        // segment's relationship sequence. The lowered form can produce an
        // equivalent trail, but `pattern_index`-keyed assignments and the
        // shortest-path rewriters read the pattern's shape directly, so this
        // pass stays out of any clause that has one.
        if !mc.path_assignments.is_empty() {
            continue;
        }
        for pattern in &mut mc.patterns {
            lower_pattern(pattern);
        }
    }
}

/// Rewrite one pattern in place. Returns whether anything changed.
fn lower_pattern(pattern: &mut Pattern) -> bool {
    let Some(total_hops) = lowered_hop_count(pattern) else {
        return false;
    };
    if total_hops > MAX_LOWERED_PATTERN_HOPS {
        return false;
    }

    let mut lowered: Vec<PatternElement> = Vec::with_capacity(pattern.elements.len() + total_hops);
    for element in &pattern.elements {
        let PatternElement::Edge(edge) = element else {
            lowered.push(element.clone());
            continue;
        };
        match lowering_hops(edge) {
            Some(k) => push_lowered_hops(&mut lowered, edge, k),
            None => lowered.push(element.clone()),
        }
    }
    pattern.elements = lowered;
    true
}

/// Total relationship elements this pattern would have after lowering, or
/// `None` when no segment is eligible (so there is nothing to rewrite).
fn lowered_hop_count(pattern: &Pattern) -> Option<usize> {
    let mut total = 0usize;
    let mut eligible = false;
    for element in &pattern.elements {
        let PatternElement::Edge(edge) = element else {
            continue;
        };
        match lowering_hops(edge) {
            Some(k) => {
                eligible = true;
                total += k;
            }
            None => total += 1,
        }
    }
    eligible.then_some(total)
}

/// Append `k` copies of `edge` as fixed hops, separated by anonymous nodes.
///
/// The intermediates carry no variable and no label — exactly what the
/// variable-length expansion constrains them to (nothing) — so they can
/// neither collide with a user variable nor leak a binding into the row.
fn push_lowered_hops(lowered: &mut Vec<PatternElement>, edge: &EdgePattern, k: usize) {
    let mut hop = edge.clone();
    hop.var_length = None;
    // Fixed hops track their trail unless a later pass proves they need not;
    // that tracking is what enforces relationship uniqueness across the
    // lowered segment.
    hop.needs_path_info = true;
    // Recomputed from connection-type metadata by `mark_skip_target_type_check`
    // against the lowered shape; never inherited from the star form.
    hop.skip_target_type_check = false;

    for i in 0..k {
        lowered.push(PatternElement::Edge(hop.clone()));
        if i + 1 < k {
            lowered.push(PatternElement::Node(anonymous_intermediate()));
        }
    }
}

/// A node pattern that constrains nothing and binds nothing.
fn anonymous_intermediate() -> NodePattern {
    NodePattern {
        variable: None,
        node_type: None,
        extra_labels: Vec::new(),
        properties: None,
        label_params: Vec::new(),
    }
}

/// How many fixed hops `edge` lowers to, or `None` when it must stay as it
/// is. The bail matrix, in order:
///
/// - not variable-length, or `min != max` — a genuine range has no fixed
///   spelling;
/// - `k == 0` — `*0..0` binds the source to the target with no relationship
///   at all, which no fixed hop expresses;
/// - `k > MAX_LOWERED_PATTERN_HOPS` — see the constant;
/// - a relationship variable is bound: `r` in `-[r:R*2..2]->` binds the
///   *list* of relationships, and lowered hops produce individual bindings,
///   not a list;
/// - a pushed-down relationship filter is already attached: this pass runs
///   before `extract_pushable_rel_predicates`, so `edge_filter` is `None` for
///   every query that reaches it through the registry; a `Some` here means
///   the pass order moved, and copying an anchor-relative predicate onto `k`
///   hops is not obviously the same predicate;
/// - an unresolved `-[:$type]->` parameter slot: those are resolved before
///   planning, so this is the same "the pipeline moved" guard.
///
/// Type constraints (`:A`, `:A|B`) and inline relationship properties are
/// *replicated* onto every hop rather than bailed on: variable-length
/// semantics require every relationship in the segment to satisfy them, which
/// is exactly what a per-hop copy asks.
fn lowering_hops(edge: &EdgePattern) -> Option<usize> {
    let (min, max) = edge.var_length?;
    if min != max || min == 0 || min > MAX_LOWERED_PATTERN_HOPS {
        return None;
    }
    if edge.variable.is_some() || edge.edge_filter.is_some() || !edge.type_params.is_empty() {
        return None;
    }
    Some(min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::pattern_matching::{parse_pattern, EdgeDirection};

    /// `(type spelling, var_length, direction)` per relationship element.
    fn hops(pattern: &Pattern) -> Vec<(Option<String>, Option<(usize, usize)>, EdgeDirection)> {
        pattern
            .elements
            .iter()
            .filter_map(|element| match element {
                PatternElement::Edge(edge) => Some((
                    edge.connection_type.clone(),
                    edge.var_length,
                    edge.direction,
                )),
                _ => None,
            })
            .collect()
    }

    fn lowered(text: &str) -> (Pattern, bool) {
        let mut pattern = parse_pattern(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        let changed = lower_pattern(&mut pattern);
        (pattern, changed)
    }

    #[test]
    fn two_hop_segment_becomes_two_fixed_hops_with_an_anonymous_intermediate() {
        let (pattern, changed) = lowered("(a:N)-[:R*2..2]->(b:N)");
        assert!(changed);
        assert_eq!(pattern.elements.len(), 5);
        assert_eq!(
            hops(&pattern),
            vec![
                (Some("R".to_string()), None, EdgeDirection::Outgoing),
                (Some("R".to_string()), None, EdgeDirection::Outgoing),
            ]
        );
        let PatternElement::Node(mid) = &pattern.elements[2] else {
            panic!("element 2 is not a node: {:?}", pattern.elements[2]);
        };
        assert!(mid.variable.is_none());
        assert!(mid.node_type.is_none());
        assert!(mid.properties.is_none());
        assert!(mid.extra_labels.is_empty());
    }

    #[test]
    fn lowered_hops_keep_trail_tracking_and_drop_the_stale_type_hint() {
        let (pattern, _) = lowered("(a:N)-[:R*3..3]->(b:N)");
        for element in &pattern.elements {
            if let PatternElement::Edge(edge) = element {
                assert!(edge.needs_path_info, "lowered hop lost its trail");
                assert!(!edge.skip_target_type_check);
            }
        }
    }

    #[test]
    fn direction_and_alternation_and_properties_replicate_onto_every_hop() {
        let (undirected, _) = lowered("(a:N)-[:R*2..2]-(b:N)");
        assert_eq!(
            hops(&undirected).iter().map(|h| h.2).collect::<Vec<_>>(),
            vec![EdgeDirection::Both, EdgeDirection::Both]
        );

        let (incoming, _) = lowered("(a:N)<-[:R*2..2]-(b:N)");
        assert_eq!(
            hops(&incoming).iter().map(|h| h.2).collect::<Vec<_>>(),
            vec![EdgeDirection::Incoming, EdgeDirection::Incoming]
        );

        let (alternation, _) = lowered("(a:N)-[:A|B*2..2]->(b:N)");
        for element in &alternation.elements {
            if let PatternElement::Edge(edge) = element {
                assert_eq!(
                    edge.connection_types.as_deref(),
                    Some(["A".to_string(), "B".to_string()].as_slice())
                );
            }
        }
    }

    #[test]
    fn ranges_zero_hops_and_bound_relationship_variables_are_left_alone() {
        for text in [
            "(a:N)-[:R*1..3]->(b:N)",
            "(a:N)-[:R*0..0]->(b:N)",
            "(a:N)-[:R*2..3]->(b:N)",
            "(a:N)-[r:R*2..2]->(b:N)",
            "(a:N)-[:R]->(b:N)",
        ] {
            let (pattern, changed) = lowered(text);
            assert!(!changed, "{text} was rewritten");
            assert_eq!(hops(&pattern).len(), 1, "{text}");
        }
    }

    #[test]
    fn the_hop_ceiling_is_eight() {
        let (eight, changed) = lowered("(a:N)-[:R*8..8]->(b:N)");
        assert!(changed);
        assert_eq!(hops(&eight).len(), 8);

        let (nine, changed) = lowered("(a:N)-[:R*9..9]->(b:N)");
        assert!(!changed);
        assert_eq!(
            hops(&nine),
            vec![(Some("R".to_string()), Some((9, 9)), EdgeDirection::Outgoing)]
        );
    }

    #[test]
    fn the_ceiling_counts_the_whole_pattern_not_one_segment() {
        // 4 + 4 fits; 4 + 5 does not, and the whole pattern then stays as written.
        let (fits, changed) = lowered("(a:N)-[:A*4..4]->(b:N)-[:B*4..4]->(c:N)");
        assert!(changed);
        assert_eq!(hops(&fits).len(), 8);

        let (over, changed) = lowered("(a:N)-[:A*4..4]->(b:N)-[:B*5..5]->(c:N)");
        assert!(!changed);
        assert_eq!(hops(&over).len(), 2);

        // Fixed hops already in the pattern count against the same budget.
        let (mixed, changed) = lowered("(a:N)-[:A]->(b:N)-[:B*2..2]->(c:N)");
        assert!(changed);
        assert_eq!(hops(&mixed).len(), 3);
    }

    #[test]
    fn a_non_lowerable_segment_does_not_block_its_neighbour() {
        let (pattern, changed) = lowered("(a:N)-[:A*2..2]->(b:N)-[:B*1..3]->(c:N)");
        assert!(changed);
        assert_eq!(
            hops(&pattern),
            vec![
                (Some("A".to_string()), None, EdgeDirection::Outgoing),
                (Some("A".to_string()), None, EdgeDirection::Outgoing),
                (Some("B".to_string()), Some((1, 3)), EdgeDirection::Outgoing),
            ]
        );
    }

    #[test]
    fn a_lowered_same_type_segment_is_never_marked_disjoint() {
        // The uniqueness opt-out is type-based, so `k` copies of one edge
        // pattern can never satisfy it — the guarantee that lowering keeps
        // trail semantics. Untyped and alternation spellings likewise.
        for text in [
            "(a:N)-[:R*2..2]->(b:N)",
            "(a:N)-[:R*2..2]-(b:N)",
            "(a:N)-[:A|B*2..2]->(b:N)",
            "(a:N)-[*2..2]->(b:N)",
        ] {
            let (pattern, changed) = lowered(text);
            assert!(changed, "{text}");
            assert!(
                !super::super::annotations::fixed_edge_types_are_pairwise_disjoint(&pattern),
                "{text} would have its trail bookkeeping removed"
            );
        }

        // ...and the `k == 1` case, where dropping the trail IS sound, still
        // reaches the marker.
        let (single, _) = lowered("(a:N)-[:R*1..1]->(b:N)");
        assert!(super::super::annotations::fixed_edge_types_are_pairwise_disjoint(&single));
    }
}
