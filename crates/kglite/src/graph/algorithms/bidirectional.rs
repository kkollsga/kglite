//! The meet-in-the-middle pair search behind `shortest_path`,
//! `shortest_path_directed` and `shortest_path_cost{,_with}`.
//!
//! Deliberately graph-agnostic: it knows only `u32` vertex ids, two neighbour
//! producers and a per-node predicate. `graph_algorithms::bidirectional_path`
//! binds it to the live graph; the split keeps the termination proof — the
//! subtle part — readable next to the loop it constrains, and testable against
//! hand-built adjacency without a `DirGraph`.

use super::Interrupt;
use rustc_hash::FxHashMap;

/// One side's BFS tree: `node -> (parent, hops from that side's root)`. The
/// root is recorded as its own parent, which is what terminates a
/// reconstruction walk.
///
/// A `HashMap` rather than two `node_bound`-sized `Vec`s, deliberately: the
/// 0.9.53 fix measured that the flat arrays cost 500 KB + 2 MB of allocate +
/// zero per call on a 500 K-node graph *regardless of how far the search
/// actually walked*, and that this alone dominated a shallow lookup (37 µs →
/// 4 µs when it went away). The Cypher `shortestPath()` executor runs one of
/// these per row, so the per-call floor is the number that matters, not the
/// per-node constant. The same reasoning is why [`bidirectional_bfs`] does not
/// borrow S3's [`BfsScratch`], whose generation stamps pay for themselves only
/// across many searches over one fixed vertex space.
type BfsTree = FxHashMap<u32, (u32, u32)>;

/// The four references one round of [`bidirectional_bfs`] works on: the
/// frontier to drain, the tree to grow, the *opposite* tree to test meetings
/// against, and the neighbour producer for this side's direction. Which side
/// they name is chosen per round, by whichever frontier is smaller.
type BidirSide<'a> = (
    &'a [u32],
    &'a mut BfsTree,
    &'a BfsTree,
    &'a mut dyn FnMut(u32, &mut dyn FnMut(u32)),
);

/// Shortest path between two anchored endpoints by **meeting in the middle**:
/// one BFS grows from `source` along the edges as given, a second grows from
/// `target` along their reverse, and the answer is stitched together where the
/// two trees first touch.
///
/// A one-sided BFS explores every node within the full path radius `d`; two
/// searches meeting in the middle explore two balls of radius `d/2`, which on
/// any graph with meaningful branching is a small fraction of the work — the
/// difference between `b^d` and `2·b^(d/2)`.
///
/// # Termination — the rule that makes the length right
///
/// A round expands **one** side, the one with the smaller frontier, through a
/// **whole level**, and stops the instant that level touches the other tree.
/// The two sides are never interleaved inside a level: a side only starts a
/// new level once the other side's last one finished.
///
/// That invariant — *each tree holds exactly the nodes within `df` (resp.
/// `db`) hops of its root* — is what makes the first touch optimal, and it is
/// worth spelling out because the plausible-looking alternatives are wrong.
/// Suppose the trees are disjoint at the top of a round (they are: any node
/// entering both would have been detected by whichever insertion came second).
/// A source→target path of length `L ≤ df + db` would put its node at offset
/// `df` in *both* trees, so disjointness proves `L ≥ df + db + 1`. Now expand
/// the forward side to `df + 1` and find `w` already in the backward tree, at
/// `dist_backward(w) = k ≤ db`. Splicing gives a real path of `df + 1 + k`, so
/// `L ≤ df + 1 + k ≤ df + db + 1 ≤ L` — every inequality collapses, and `w` is
/// on a shortest path whichever `w` the level happened to reach first. Because
/// the round returns immediately, the half-drained level is never used for
/// anything.
///
/// What breaks the argument is advancing one side while the other's level is
/// half-expanded — the textbook shape being "expand both frontiers a level,
/// then compare, and report `forward_level + backward_level`". That variant is
/// right on even-length shortest paths and one hop too long on odd-length
/// ones, where the two halves meet on the middle *edge* and a per-round
/// counter cannot express the half-round.
/// `test_bidirectional_beats_the_naive_level_counter_off_by_one` pins the
/// asymmetry against a deliberately-naive reference, and
/// `test_bidirectional_matches_one_sided_on_random_graphs` cross-checks every
/// length against a one-sided BFS over thousands of random pairs.
///
/// # Arguments
///
/// * `expand_fwd(u, sink)` feeds `sink` every node reachable from `u` in the
///   query's direction; `expand_bwd(u, sink)` does the same for the reverse
///   direction. Duplicates are harmless — the trees double as visited sets.
/// * `via(w)` gates a node's use as an **intermediate** hop, identically on
///   both frontiers. `source` and `target` are exempt (checked here, so a
///   caller's predicate need not special-case them), matching the rest of the
///   family: `via_types` restricts the middle of a path, never its ends.
/// * `deadline` is polled every 1000 frontier nodes; expiry answers `None`,
///   the same "no answer" a disconnected pair gets.
///
/// Returns the node sequence from `source` to `target` inclusive, or `None`.
/// When several shortest paths tie, *which* one comes back is arbitrary — as
/// it always was for the one-sided search, but a different arbitrary choice.
pub(super) fn bidirectional_bfs(
    source: u32,
    target: u32,
    mut expand_fwd: impl FnMut(u32, &mut dyn FnMut(u32)),
    mut expand_bwd: impl FnMut(u32, &mut dyn FnMut(u32)),
    via: impl Fn(u32) -> bool,
    deadline: Interrupt,
) -> Option<Vec<u32>> {
    if source == target {
        return Some(vec![source]);
    }

    let mut tree_f = BfsTree::default();
    let mut tree_b = BfsTree::default();
    tree_f.insert(source, (source, 0));
    tree_b.insert(target, (target, 0));

    let mut frontier_f: Vec<u32> = vec![source];
    let mut frontier_b: Vec<u32> = vec![target];
    let mut next: Vec<u32> = Vec::new();
    let mut visits = 0u32;

    // Either frontier running dry means that side's component is exhausted:
    // no path exists.
    while !frontier_f.is_empty() && !frontier_b.is_empty() {
        // Expand the cheaper side. On a graph whose in- and out-degrees differ
        // wildly this is what keeps the search balanced by *work* rather than
        // by hop count.
        let forward = frontier_f.len() <= frontier_b.len();
        let (frontier, tree, other, expand): BidirSide = if forward {
            (&frontier_f, &mut tree_f, &tree_b, &mut expand_fwd)
        } else {
            (&frontier_b, &mut tree_b, &tree_f, &mut expand_bwd)
        };

        next.clear();
        // First (node, total hops) meeting this level produces; see the
        // termination proof above for why the first is already optimal.
        let mut meet: Option<(u32, u32)> = None;

        for &u in frontier {
            visits = visits.wrapping_add(1);
            if visits.is_multiple_of(1000) && deadline.exceeded() {
                return None;
            }
            let du = tree[&u].1;
            expand(u, &mut |w| {
                if meet.is_some() {
                    return;
                }
                if tree.contains_key(&w) {
                    return;
                }
                if w != source && w != target && !via(w) {
                    return;
                }
                tree.insert(w, (u, du + 1));
                next.push(w);
                if let Some(&(_, dw)) = other.get(&w) {
                    meet = Some((w, du + 1 + dw));
                }
            });
            if meet.is_some() {
                break;
            }
        }

        if let Some((node, total)) = meet {
            let path = stitch_path(&tree_f, &tree_b, node);
            // The two half-walks must reproduce the distance the meeting
            // reported; a stitch that dropped or doubled the meeting node
            // would still look like a path without this.
            debug_assert_eq!(path.len() - 1, total as usize);
            return Some(path);
        }

        if forward {
            std::mem::swap(&mut frontier_f, &mut next);
        } else {
            std::mem::swap(&mut frontier_b, &mut next);
        }
    }

    None
}

/// Splice the two half-paths meeting at `node` into one `source → target`
/// sequence: the forward tree walked back to its root and reversed, then the
/// backward tree walked forward to *its* root, which is the target.
fn stitch_path(tree_f: &BfsTree, tree_b: &BfsTree, node: u32) -> Vec<u32> {
    let mut path = Vec::with_capacity(16);
    let mut cursor = node;
    loop {
        path.push(cursor);
        let (parent, _) = tree_f[&cursor];
        if parent == cursor {
            break;
        }
        cursor = parent;
    }
    path.reverse();

    let mut cursor = node;
    loop {
        let (parent, _) = tree_b[&cursor];
        if parent == cursor {
            break;
        }
        path.push(parent);
        cursor = parent;
    }
    path
}
