//! The single ORDER BY comparison used by every sorting path.
//!
//! Sort order is defined exactly once here — [`compare_sort_keys`] — and is
//! shared by the full sort ([`super::CypherExecutor::execute_order_by`]), the
//! streaming top-K operator ([`super::stream::heap_top_k`]) and both fused
//! top-K executors (`FusedOrderByTopK`, `FusedNodeScanTopK`). Before 0.15.14
//! each of those carried its own comparison, and the fused ones disagreed with
//! the full sort on NULL keys (they dropped NULL-keyed rows entirely, so
//! `ORDER BY x DESC LIMIT k` returned the wrong rows — or fewer than `k`).
//!
//! [`TopKCollector`] is the heap those top-K paths share: it keeps at most `K`
//! entries ranked by `compare_sort_keys`, so a fused plan and the unfused
//! `ORDER BY` + `LIMIT` pipeline cannot drift apart.

use super::super::ast::{NullsPlacement, OrderItem};
use crate::datatypes::values::Value;
use std::cmp::Ordering;

/// Per-key sort spec: direction plus the *resolved* NULLS placement
/// (explicit `NULLS FIRST/LAST` if written, else ASC → Last, DESC → First —
/// the Neo4j 5+ default, 0.9.0 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SortSpec {
    pub(crate) ascending: bool,
    pub(crate) nulls: NullsPlacement,
}

impl SortSpec {
    pub(crate) fn from_order_item(item: &OrderItem) -> Self {
        SortSpec {
            ascending: item.ascending,
            nulls: item.effective_nulls(),
        }
    }
}

/// Lexicographic ORDER BY comparison over a key tuple, *better first*:
/// `Less` means `a` sorts before `b` in the emitted result.
///
/// Per key, in order: NULLs are placed by the key's `NullsPlacement`
/// (overriding `compare_values`, which sorts NULL below everything);
/// otherwise the values are compared and the result reversed for DESC.
/// Keys that are incomparable (`compare_values` → `None`, e.g. a string
/// against a list) or equal fall through to the next key; a tuple that ties on
/// every key compares `Equal`, leaving the caller's stable order intact.
pub(crate) fn compare_sort_keys(a: &[Value], b: &[Value], specs: &[SortSpec]) -> Ordering {
    for (i, spec) in specs.iter().enumerate() {
        let key_a = a.get(i).unwrap_or(&Value::Null);
        let key_b = b.get(i).unwrap_or(&Value::Null);

        let a_null = matches!(key_a, Value::Null);
        let b_null = matches!(key_b, Value::Null);
        match (a_null, b_null) {
            (true, true) => continue,
            (true, false) => {
                return match spec.nulls {
                    NullsPlacement::First => Ordering::Less,
                    NullsPlacement::Last => Ordering::Greater,
                };
            }
            (false, true) => {
                return match spec.nulls {
                    NullsPlacement::First => Ordering::Greater,
                    NullsPlacement::Last => Ordering::Less,
                };
            }
            (false, false) => {}
        }

        if let Some(ordering) = compare_one(key_a, key_b) {
            let oriented = if spec.ascending {
                ordering
            } else {
                ordering.reverse()
            };
            if oriented != Ordering::Equal {
                return oriented;
            }
        }
    }
    Ordering::Equal
}

/// One key comparison. The three arms here are the same-type cases of
/// [`crate::graph::core::filtering::compare_values`], repeated only so they
/// inline: every other pair — cross-type numerics, dates, strings parsed as
/// dates, incomparable types — falls through to that function, which remains
/// the definition. Ordering semantics live in one place; this is a call-site
/// shortcut, not a second rule set.
#[inline]
fn compare_one(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y),
        (Value::Int64(x), Value::Int64(y)) => Some(x.cmp(y)),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => crate::graph::core::filtering::compare_values(a, b),
    }
}

/// Direction-folded `f64` stand-in for the *first* sort key, or NaN when the
/// key has no such stand-in (NULL, string, boolean, date, an actual NaN, or no
/// key at all). `sign` is `-1.0` for a DESC first key, so a plain `partial_cmp`
/// of two lanes is already better-first.
///
/// Only a strict Less/Greater from the lane is trusted; Equal and NaN hand the
/// pair back to [`compare_sort_keys`]. That keeps the lane *exact*: `i64 → f64`
/// is monotone (round-to-nearest never inverts an ordering), so it can never
/// disagree about direction — it can only lose resolution above 2^53, and that
/// shows up as Equal, which delegates. NULL placement, cross-type rules,
/// later keys and the `seq` tiebreak are therefore all still decided by the
/// one comparator.
#[inline]
fn fast_lane(keys: &[Value], sign: f64) -> f64 {
    match keys.first() {
        Some(Value::Float64(f)) => sign * f,
        Some(Value::Int64(i)) => sign * (*i as f64),
        Some(Value::UniqueId(u)) => sign * (*u as f64),
        _ => f64::NAN,
    }
}

/// One retained candidate: its sort-key tuple, its input position (`seq`), its
/// [`fast_lane`] stand-in and the caller's payload.
struct Entry<P> {
    keys: Vec<Value>,
    seq: usize,
    lane: f64,
    payload: P,
}

/// Bounded top-K heap: retains the `limit` best entries under
/// [`compare_sort_keys`], in O(n log k) time and O(k) memory.
///
/// The heap is kept by hand rather than through `BinaryHeap<Entry>` for one
/// measured reason: an *improving* key stream — `ORDER BY value DESC` over a
/// column that ascends with scan order, i.e. a leaderboard over an
/// append-ordered table — beats the current worst on **every** row, so the
/// retention path runs n times, not k, and its per-row cost is the query's
/// cost. Through `BinaryHeap` each retention allocated a fresh key `Vec`,
/// dropped the evicted one, and bumped an `Arc<[SortSpec]>` refcount (the specs
/// had to ride inside every entry, because `Ord` sees only the entries).
/// Replacing the root in place reuses the evicted entry's key buffer and holds
/// the specs once on the collector; the [`fast_lane`] shortcut then takes the
/// comparisons themselves off the `Value` dispatch. `ordering::tests::
/// top_k_retention_cost` reproduces the decomposition.
pub(crate) struct TopKCollector<P> {
    limit: usize,
    specs: Vec<SortSpec>,
    /// `-1.0` when the first key is DESC — see [`fast_lane`].
    sign: f64,
    /// Max-heap by *worst* entry: `heap[0]` is the first candidate to evict.
    heap: Vec<Entry<P>>,
}

impl<P> TopKCollector<P> {
    pub(crate) fn new(specs: Vec<SortSpec>, limit: usize) -> Self {
        // No specs means "every tuple ties, keep input order" — a NaN sign
        // disables the lane so that stays true.
        let sign = match specs.first() {
            Some(spec) if spec.ascending => 1.0,
            Some(_) => -1.0,
            None => f64::NAN,
        };
        TopKCollector {
            limit,
            specs,
            sign,
            heap: Vec::with_capacity(limit.min(1024)),
        }
    }

    /// Better-first rank of two retained entries. Ties on every key break by
    /// input position, so the retained set and its order match a *stable* full
    /// sort exactly.
    #[inline]
    fn rank(&self, a: &Entry<P>, b: &Entry<P>) -> Ordering {
        match a.lane.partial_cmp(&b.lane) {
            Some(Ordering::Less) => Ordering::Less,
            Some(Ordering::Greater) => Ordering::Greater,
            _ => compare_sort_keys(&a.keys, &b.keys, &self.specs).then_with(|| a.seq.cmp(&b.seq)),
        }
    }

    /// Better-first rank of a not-yet-retained candidate against an entry.
    #[inline]
    fn rank_candidate(&self, keys: &[Value], lane: f64, seq: usize, other: &Entry<P>) -> Ordering {
        match lane.partial_cmp(&other.lane) {
            Some(Ordering::Less) => Ordering::Less,
            Some(Ordering::Greater) => Ordering::Greater,
            _ => {
                compare_sort_keys(keys, &other.keys, &self.specs).then_with(|| seq.cmp(&other.seq))
            }
        }
    }

    /// Would `keys` at input position `seq` enter the current top-K?
    ///
    /// Purely a work guard — [`push`](Self::push) is correct on its own and
    /// re-checks, dropping a candidate that does not beat the worst retained
    /// entry. Callers evaluate sort keys into a reusable buffer and only pay
    /// for cloning the tuple when this returns true.
    pub(crate) fn accepts(&self, keys: &[Value], seq: usize) -> bool {
        if self.limit == 0 {
            return false;
        }
        if self.heap.len() < self.limit {
            return true;
        }
        match self.heap.first() {
            Some(root) => {
                self.rank_candidate(keys, fast_lane(keys, self.sign), seq, root) == Ordering::Less
            }
            None => true,
        }
    }

    /// Offer a candidate. Below capacity it is retained; at capacity it
    /// replaces the worst retained entry if it ranks better, reusing that
    /// entry's key buffer, and is dropped otherwise.
    pub(crate) fn push(&mut self, keys: &[Value], seq: usize, payload: P) {
        if self.limit == 0 {
            return;
        }
        let lane = fast_lane(keys, self.sign);
        if self.heap.len() < self.limit {
            self.heap.push(Entry {
                keys: keys.to_vec(),
                seq,
                lane,
                payload,
            });
            self.sift_up(self.heap.len() - 1);
            return;
        }
        if self.rank_candidate(keys, lane, seq, &self.heap[0]) != Ordering::Less {
            return;
        }
        let root = &mut self.heap[0];
        root.keys.clear();
        root.keys.extend_from_slice(keys);
        root.seq = seq;
        root.lane = lane;
        root.payload = payload;
        self.sift_down(0);
    }

    /// Restore the heap upward from `idx`: an entry worse than its parent
    /// rises toward the root.
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.rank(&self.heap[idx], &self.heap[parent]) != Ordering::Greater {
                break;
            }
            self.heap.swap(idx, parent);
            idx = parent;
        }
    }

    /// Restore the heap downward from `idx`: an entry better than a child
    /// sinks, so the worst retained entry stays at the root.
    fn sift_down(&mut self, mut idx: usize) {
        let len = self.heap.len();
        loop {
            let (left, right) = (2 * idx + 1, 2 * idx + 2);
            let mut worst = idx;
            if left < len && self.rank(&self.heap[left], &self.heap[worst]) == Ordering::Greater {
                worst = left;
            }
            if right < len && self.rank(&self.heap[right], &self.heap[worst]) == Ordering::Greater {
                worst = right;
            }
            if worst == idx {
                return;
            }
            self.heap.swap(idx, worst);
            idx = worst;
        }
    }

    /// Drain into result order — best first — as `(sort keys, payload)` pairs.
    /// The keys come back so callers can reuse them for RETURN items that *are*
    /// the sort key instead of re-evaluating the expression.
    pub(crate) fn into_sorted(mut self) -> Vec<(Vec<Value>, P)> {
        let specs = std::mem::take(&mut self.specs);
        self.heap.sort_by(|a, b| {
            compare_sort_keys(&a.keys, &b.keys, &specs).then_with(|| a.seq.cmp(&b.seq))
        });
        self.heap
            .into_iter()
            .map(|entry| (entry.keys, entry.payload))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asc() -> SortSpec {
        SortSpec {
            ascending: true,
            nulls: NullsPlacement::Last,
        }
    }

    fn desc() -> SortSpec {
        SortSpec {
            ascending: false,
            nulls: NullsPlacement::First,
        }
    }

    /// Reference implementation: stable full sort by the same comparator.
    fn full_sort(rows: &[Vec<Value>], specs: &[SortSpec], limit: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| compare_sort_keys(&rows[a], &rows[b], specs));
        idx.truncate(limit);
        idx
    }

    fn collect_top_k(rows: &[Vec<Value>], specs: &[SortSpec], limit: usize) -> Vec<usize> {
        let mut collector: TopKCollector<usize> = TopKCollector::new(specs.to_vec(), limit);
        for (seq, keys) in rows.iter().enumerate() {
            if collector.accepts(keys, seq) {
                collector.push(keys, seq, seq);
            }
        }
        collector
            .into_sorted()
            .into_iter()
            .map(|(_, payload)| payload)
            .collect()
    }

    /// Retention that allocates a fresh key tuple per replacement instead of
    /// reusing the evicted entry's buffer — the isolate for the allocation
    /// half of the [`top_k_retention_cost`] decomposition. Test-only.
    fn push_no_reuse(c: &mut TopKCollector<usize>, keys: &[Value], seq: usize, payload: usize) {
        if c.heap.len() < c.limit {
            c.push(keys, seq, payload);
            return;
        }
        let lane = fast_lane(keys, c.sign);
        if c.rank_candidate(keys, lane, seq, &c.heap[0]) != Ordering::Less {
            return;
        }
        c.heap[0] = Entry {
            keys: keys.to_vec(),
            seq,
            lane,
            payload,
        };
        c.sift_down(0);
    }

    /// min-of-rounds wall time for `f`, after 3 warmup rounds.
    fn best<R>(rounds: usize, mut f: impl FnMut() -> R) -> std::time::Duration {
        use std::time::Instant;
        for _ in 0..3 {
            std::hint::black_box(f());
        }
        let mut best = std::time::Duration::MAX;
        for _ in 0..rounds {
            let t = Instant::now();
            std::hint::black_box(f());
            let d = t.elapsed();
            if d < best {
                best = d;
            }
        }
        best
    }

    /// Not a gate — the reproducible decomposition behind `TopKCollector`'s
    /// shape. Run with `cargo test --release -p kglite --lib
    /// top_k_retention_cost -- --ignored --nocapture`; debug-profile numbers
    /// are meaningless. Reports min-of-30 for a 50k-row scan at k=10: an
    /// improving DESC stream retains every row, so its per-row retention cost
    /// is the whole cost, while ASC and the string column retain a handful.
    #[test]
    #[ignore]
    fn top_k_retention_cost() {
        let n = 50_000usize;
        let numeric: Vec<Value> = (0..n).map(|i| Value::Float64(i as f64)).collect();
        let strings: Vec<Value> = (0..n)
            .map(|i| Value::String(format!("hc_{}", i % (n / 2))))
            .collect();

        let cmp = best(30, || {
            let mut sink = 0usize;
            for i in 0..n {
                if compare_sort_keys(
                    std::slice::from_ref(&numeric[i]),
                    std::slice::from_ref(&numeric[(i + 7) % n]),
                    &[desc()],
                ) == Ordering::Less
                {
                    sink += 1;
                }
            }
            sink
        });
        println!("compare_sort_keys x{n} (Float64):  min={cmp:>10.3?}");

        for (label, col) in [("numeric", &numeric), ("string", &strings)] {
            // Control: the caller-side loop with no collector at all.
            let ctrl = best(30, || {
                let mut buf: Vec<Value> = Vec::with_capacity(1);
                let mut sink = 0usize;
                for v in col.iter() {
                    buf.clear();
                    buf.push(v.clone());
                    sink += buf.len();
                }
                sink
            });
            println!("{label:8} loop-only            min={ctrl:>10.3?}");

            for (dir, spec) in [("asc", asc()), ("desc", desc())] {
                let mut accepted = 0usize;
                let full = best(30, || {
                    let mut buf: Vec<Value> = Vec::with_capacity(1);
                    accepted = 0;
                    let mut c: TopKCollector<usize> = TopKCollector::new(vec![spec], 10);
                    for (seq, v) in col.iter().enumerate() {
                        buf.clear();
                        buf.push(v.clone());
                        if c.accepts(&buf, seq) {
                            accepted += 1;
                            c.push(&buf, seq, seq);
                        }
                    }
                    c.into_sorted().len()
                });
                let no_reuse = best(30, || {
                    let mut buf: Vec<Value> = Vec::with_capacity(1);
                    let mut d: TopKCollector<usize> = TopKCollector::new(vec![spec], 10);
                    for (seq, v) in col.iter().enumerate() {
                        buf.clear();
                        buf.push(v.clone());
                        if d.accepts(&buf, seq) {
                            push_no_reuse(&mut d, &buf, seq, seq);
                        }
                    }
                    d.into_sorted().len()
                });
                println!(
                    "{label:8} {dir:5} reuse={full:>10.3?} fresh_alloc={no_reuse:>10.3?} \
                     accepted={accepted:6}"
                );
            }
        }
    }

    /// The numeric fast lane may only ever *shortcut* the comparator, never
    /// answer differently. These rows are built to hit every way it can lose
    /// information: integers past 2^53 that collapse onto one `f64`, mixed
    /// Int/Float/UniqueId keys in one column, NULLs, and heavy duplication so
    /// the `seq` tiebreak decides.
    ///
    /// The first key stays *mutually comparable* on purpose. Mixing, say, a
    /// string into a numeric key column makes `compare_sort_keys` intransitive
    /// (incomparable pairs fall through to the next key), and an intransitive
    /// comparator lets any heap disagree with any stable sort — a pre-existing
    /// property of the mixed-type ordering rules, unrelated to the lane, which
    /// hands every string/date/bool key straight back to the comparator.
    #[test]
    fn fast_lane_never_disagrees_with_the_full_sort() {
        const BIG: i64 = (1i64 << 53) + 1;
        let rows: Vec<Vec<Value>> = (0..400)
            .map(|i| {
                let key0 = match i % 6 {
                    0 => Value::Int64(BIG + (i as i64 % 3)),
                    1 => Value::Float64((i % 5) as f64),
                    2 => Value::Int64((i % 5) as i64),
                    3 => Value::UniqueId((i % 5) as u32),
                    4 => Value::Null,
                    _ => Value::Float64(f64::from(-(i % 4))),
                };
                vec![key0, Value::Int64((i % 11) as i64)]
            })
            .collect();

        for specs in [
            vec![asc()],
            vec![desc()],
            vec![asc(), desc()],
            vec![desc(), asc()],
            vec![
                SortSpec {
                    ascending: false,
                    nulls: NullsPlacement::Last,
                },
                asc(),
            ],
        ] {
            for limit in [1usize, 3, 17, 400] {
                assert_eq!(
                    collect_top_k(&rows, &specs, limit),
                    full_sort(&rows, &specs, limit),
                    "fast lane diverged from full sort at limit {limit} for {specs:?}"
                );
            }
        }
    }

    #[test]
    fn nulls_place_by_spec_not_by_compare_values() {
        let specs = [desc()];
        // DESC defaults to NULLS FIRST, so NULL outranks every number.
        assert_eq!(
            compare_sort_keys(&[Value::Null], &[Value::Int64(9)], &specs),
            Ordering::Less
        );
        let specs = [asc()];
        assert_eq!(
            compare_sort_keys(&[Value::Null], &[Value::Int64(9)], &specs),
            Ordering::Greater
        );
    }

    #[test]
    fn later_keys_break_ties_independently_of_direction() {
        let specs = [desc(), asc()];
        let a = vec![Value::Int64(1), Value::Int64(5)];
        let b = vec![Value::Int64(1), Value::Int64(7)];
        assert_eq!(compare_sort_keys(&a, &b, &specs), Ordering::Less);
        let specs = [desc(), desc()];
        assert_eq!(compare_sort_keys(&a, &b, &specs), Ordering::Greater);
    }

    #[test]
    fn incomparable_keys_fall_through_to_the_next_key() {
        let specs = [asc(), asc()];
        let a = vec![Value::String("x".into()), Value::Int64(1)];
        let b = vec![Value::Boolean(true), Value::Int64(2)];
        assert_eq!(compare_sort_keys(&a, &b, &specs), Ordering::Less);
    }

    /// The whole point of the shared comparator: the bounded heap and a stable
    /// full sort must select and order the same rows, including ties and NULLs.
    #[test]
    fn top_k_matches_a_stable_full_sort() {
        // Deterministic xorshift — no dev-dependency needed.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let rows: Vec<Vec<Value>> = (0..500)
            .map(|_| {
                let a = next() % 7;
                let b = next() % 5;
                vec![
                    if a == 0 {
                        Value::Null
                    } else {
                        Value::Int64(a as i64)
                    },
                    if b == 0 {
                        Value::Null
                    } else {
                        Value::String(format!("s{b}"))
                    },
                ]
            })
            .collect();

        for specs in [
            vec![asc(), asc()],
            vec![desc(), asc()],
            vec![asc(), desc()],
            vec![desc(), desc()],
            vec![
                SortSpec {
                    ascending: true,
                    nulls: NullsPlacement::First,
                },
                SortSpec {
                    ascending: false,
                    nulls: NullsPlacement::Last,
                },
            ],
        ] {
            for limit in [0usize, 1, 3, 25, 500, 600] {
                assert_eq!(
                    collect_top_k(&rows, &specs, limit),
                    full_sort(&rows, &specs, limit),
                    "top-K diverged from full sort at limit {limit} for {specs:?}"
                );
            }
        }
    }
}
