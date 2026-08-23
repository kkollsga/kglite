//! The single ORDER BY comparison used by every sorting path.
//!
//! Sort order is defined exactly once here — [`compare_sort_keys`] — and is
//! shared by the full sort ([`super::CypherExecutor::execute_order_by`]), the
//! streaming top-K operator ([`super::stream::heap_top_k`]) and both fused
//! top-K executors (`FusedOrderByTopK`, `FusedNodeScanTopK`), so no path can
//! disagree with another on NULL keys.
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
/// (overriding the total order, which ranks NULL last ascending); otherwise
/// the values are ranked by
/// [`total_order`](crate::graph::core::filtering::total_order) and the result
/// reversed for DESC. Keys that compare equal — including two values of
/// different types that the type rank cannot separate — fall through to the
/// next key; a tuple that ties on every key compares `Equal`, leaving the
/// caller's stable order intact.
///
/// **The comparison is total.** A key column holding more than one type
/// (a `CASE` returning a number on some rows and a string on others, a
/// property read across two node types, `coalesce` over differently-typed
/// fields) orders by type rank first, so no pair is ever "incomparable".
/// Before the total order such a pair was *skipped*, which made the comparator
/// intransitive: string-vs-number reported `Equal` while number-vs-number
/// ordered. `slice::sort_by` detects exactly that and panics
/// ("user-provided comparison function does not correctly implement a total
/// order"), aborting the query — a `PanicException` through pyo3 and an
/// unwind with no `catch_unwind` in the Bolt server. The top-K heap has no
/// such check and silently disagreed with the full sort instead.
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

        let ordering = compare_one(key_a, key_b);
        let oriented = if spec.ascending {
            ordering
        } else {
            ordering.reverse()
        };
        if oriented != Ordering::Equal {
            return oriented;
        }
    }
    Ordering::Equal
}

/// One key comparison. The three arms here are the same-type cases of
/// [`crate::graph::core::filtering::total_order`], repeated only so they
/// inline: every other pair — cross-type numerics, temporals, cross-*type*
/// pairs ranked by type — falls through to that function, which remains the
/// definition. Ordering semantics live in one place; this is a call-site
/// shortcut, not a second rule set, and
/// `ordering::tests::the_inline_shortcut_agrees_with_the_total_order` pins
/// that.
#[inline]
fn compare_one(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Float64(x), Value::Float64(y)) => {
            crate::graph::core::filtering::cmp_f64_total(*x, *y)
        }
        (Value::Int64(x), Value::Int64(y)) => x.cmp(y),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        _ => crate::graph::core::filtering::total_order(a, b),
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
    /// The first key column deliberately mixes a **string** in among the
    /// numbers. Before the total order that was impossible to test: an
    /// incomparable pair fell through to the next key, which made
    /// `compare_sort_keys` intransitive, and an intransitive comparator lets
    /// any heap disagree with any stable sort. Now the string is ranked by
    /// type (rank 9, below every number) and the lane — which hands every
    /// string/date/bool key straight back to the comparator as NaN — must
    /// still agree with the full sort.
    #[test]
    fn fast_lane_never_disagrees_with_the_full_sort() {
        const BIG: i64 = (1i64 << 53) + 1;
        let rows: Vec<Vec<Value>> = (0..400)
            .map(|i| {
                let key0 = match i % 7 {
                    0 => Value::Int64(BIG + (i as i64 % 3)),
                    1 => Value::Float64((i % 5) as f64),
                    2 => Value::Int64((i % 5) as i64),
                    3 => Value::UniqueId((i % 5) as u32),
                    4 => Value::Null,
                    5 => Value::String(format!("k{}", i % 4)),
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

    /// A cross-type first key is decided *by the type rank*, so the second key
    /// never runs. (Before the total order the same assertion held for the
    /// opposite reason: the string/boolean pair was skipped and `1 < 2`
    /// decided it — which is precisely the intransitivity that aborted sorts.)
    #[test]
    fn a_cross_type_key_is_decided_by_the_type_rank() {
        let specs = [asc(), asc()];
        // String (rank 9) before Boolean (rank 10) — the trailing key would
        // have said the opposite.
        let a = vec![Value::String("x".into()), Value::Int64(9)];
        let b = vec![Value::Boolean(true), Value::Int64(1)];
        assert_eq!(compare_sort_keys(&a, &b, &specs), Ordering::Less);
        let specs = [desc(), asc()];
        assert_eq!(compare_sort_keys(&a, &b, &specs), Ordering::Greater);
    }

    /// Every value of every rank class, in ascending total order. Ordering one
    /// of these against any other must reproduce this sequence exactly.
    fn one_of_every_rank_class() -> Vec<Value> {
        use crate::datatypes::values::{NodeValue, PathValue, RelValue};
        let node = NodeValue {
            id: 1,
            labels: vec!["N".into()],
            properties: crate::datatypes::PropMap::new(),
        };
        let rel = RelValue {
            id: 1,
            start_id: 1,
            end_id: 2,
            rel_type: "R".into(),
            properties: crate::datatypes::PropMap::new(),
        };
        vec![
            Value::Map(crate::datatypes::PropMap::from_iter([(
                "k",
                Value::Int64(1),
            )])),
            Value::Node(Box::new(node.clone())),
            Value::NodeRef(3),
            Value::Relationship(Box::new(rel.clone())),
            Value::List(vec![Value::Int64(1)]),
            Value::Path(Box::new(PathValue {
                nodes: vec![node],
                rels: vec![rel],
            })),
            Value::DateTime(chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
            Value::Timestamp(
                chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .and_hms_opt(12, 0, 0)
                    .unwrap(),
            ),
            Value::Duration {
                months: 0,
                days: 1,
                seconds: 0,
            },
            Value::Point { lat: 1.0, lon: 2.0 },
            Value::String("s".into()),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Float64(-1.5),
            Value::Int64(0),
            Value::UniqueId(1),
            Value::Float64(1.5),
            Value::Float64(f64::NAN),
            Value::Null,
        ]
    }

    /// The rank table itself: sorting the one-per-class sample reproduces the
    /// documented ascending sequence, and DESC is its exact reverse.
    #[test]
    fn the_type_rank_orders_every_value_class() {
        let expected = one_of_every_rank_class();
        let rows: Vec<Vec<Value>> = expected.iter().cloned().map(|v| vec![v]).collect();

        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.shuffle_deterministically();
        idx.sort_by(|&a, &b| {
            compare_sort_keys(
                &rows[a],
                &rows[b],
                &[SortSpec {
                    ascending: true,
                    // NULL's own rank is last ascending; assert it directly
                    // rather than through the clause default.
                    nulls: NullsPlacement::Last,
                }],
            )
        });
        let sorted: Vec<Value> = idx.iter().map(|&i| expected[i].clone()).collect();
        assert_eq!(
            format!("{sorted:?}"),
            format!("{expected:?}"),
            "ascending total order does not match the documented rank table"
        );

        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.shuffle_deterministically();
        idx.sort_by(|&a, &b| {
            compare_sort_keys(
                &rows[a],
                &rows[b],
                &[SortSpec {
                    ascending: false,
                    nulls: NullsPlacement::First,
                }],
            )
        });
        let sorted: Vec<Value> = idx.iter().map(|&i| expected[i].clone()).collect();
        let mut reversed = expected.clone();
        reversed.reverse();
        assert_eq!(
            format!("{sorted:?}"),
            format!("{reversed:?}"),
            "descending order is not the reverse of ascending"
        );
    }

    /// Deterministic shuffle so the sort has real work to do.
    trait ShuffleDeterministically {
        fn shuffle_deterministically(&mut self);
    }
    impl ShuffleDeterministically for Vec<usize> {
        fn shuffle_deterministically(&mut self) {
            let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
            for i in (1..self.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                self.swap(i, (state % (i as u64 + 1)) as usize);
            }
        }
    }

    /// Totality, proved directly rather than through a sort's internal check:
    /// antisymmetry and transitivity over every pair and triple of the sample.
    #[test]
    fn the_total_order_is_antisymmetric_and_transitive() {
        use crate::graph::core::filtering::total_order;
        let mut values = one_of_every_rank_class();
        // Duplicates and near-misses: equal-ranked values that must tie, and
        // the 2^53 neighbourhood where an `as f64` comparison stops being
        // transitive.
        values.extend([
            Value::Int64(0),
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::UniqueId(0),
            Value::Int64((1i64 << 53) + 1),
            Value::Int64(1i64 << 53),
            Value::Float64((1u64 << 53) as f64),
            Value::Float64(f64::INFINITY),
            Value::Float64(f64::NEG_INFINITY),
            Value::String(String::new()),
            Value::List(vec![]),
            Value::List(vec![Value::Int64(1), Value::String("a".into())]),
        ]);

        for a in &values {
            assert_eq!(total_order(a, a), Ordering::Equal, "not reflexive: {a:?}");
            for b in &values {
                assert_eq!(
                    total_order(a, b),
                    total_order(b, a).reverse(),
                    "not antisymmetric: {a:?} vs {b:?}"
                );
            }
        }
        for a in &values {
            for b in &values {
                let ab = total_order(a, b);
                if ab == Ordering::Greater {
                    continue;
                }
                for c in &values {
                    let bc = total_order(b, c);
                    if bc == Ordering::Greater {
                        continue;
                    }
                    // a <= b <= c  ⇒  a <= c, and a == b == c ⇒ a == c.
                    let ac = total_order(a, c);
                    assert_ne!(ac, Ordering::Greater, "not transitive: {a:?} {b:?} {c:?}");
                    if ab == Ordering::Equal && bc == Ordering::Equal {
                        assert_eq!(
                            ac,
                            Ordering::Equal,
                            "equality not transitive: {a:?} {b:?} {c:?}"
                        );
                    }
                }
            }
        }
    }

    /// The inline shortcut in [`compare_one`] must be a shortcut, not a second
    /// rule set.
    #[test]
    fn the_inline_shortcut_agrees_with_the_total_order() {
        use crate::graph::core::filtering::total_order;
        let mut values = one_of_every_rank_class();
        values.extend([
            Value::Int64(-7),
            Value::Float64(f64::NAN),
            Value::String("s".into()),
            Value::String("t".into()),
            Value::Float64(1.5),
        ]);
        for a in &values {
            for b in &values {
                assert_eq!(
                    compare_one(a, b),
                    total_order(a, b),
                    "shortcut disagrees for {a:?} vs {b:?}"
                );
            }
        }
    }

    /// Numbers order numerically across `Int64`/`Float64`/`UniqueId`, exactly
    /// — including past 2^53, where an `as f64` conversion collapses two
    /// distinct integers onto one float and makes the comparator intransitive.
    #[test]
    fn integers_past_2_pow_53_compare_exactly_against_floats() {
        use crate::graph::core::filtering::total_order;
        const BIG: i64 = (1i64 << 53) + 1;
        let float = Value::Float64((1u64 << 53) as f64);
        assert_eq!(total_order(&Value::Int64(BIG), &float), Ordering::Greater);
        assert_eq!(
            total_order(&Value::Int64(1i64 << 53), &float),
            Ordering::Equal
        );
        assert_eq!(total_order(&float, &Value::Int64(BIG)), Ordering::Less);
        // Fractions and range extremes.
        assert_eq!(
            total_order(&Value::Int64(2), &Value::Float64(2.5)),
            Ordering::Less
        );
        assert_eq!(
            total_order(&Value::Int64(-2), &Value::Float64(-2.5)),
            Ordering::Greater
        );
        assert_eq!(
            total_order(&Value::Int64(i64::MAX), &Value::Float64(f64::INFINITY)),
            Ordering::Less
        );
        assert_eq!(
            total_order(&Value::Int64(i64::MIN), &Value::Float64(f64::NEG_INFINITY)),
            Ordering::Greater
        );
        // NaN sorts above every number.
        assert_eq!(
            total_order(&Value::Int64(i64::MAX), &Value::Float64(f64::NAN)),
            Ordering::Less
        );
        assert_eq!(
            total_order(&Value::UniqueId(3), &Value::Int64(3)),
            Ordering::Equal
        );
    }

    /// Deterministic xorshift stream of `n` rows whose single sort key is a
    /// string half the time and an integer the other half — the shape that
    /// used to abort `ORDER BY` (see [`mixed_type_column_sorts_without_panicking`]).
    fn mixed_rows(n: usize) -> Vec<Vec<Value>> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|_| {
                let key = if next() % 2 == 0 {
                    Value::Int64((next() % 7) as i64)
                } else {
                    Value::String(format!("s{}", next() % 7))
                };
                vec![key]
            })
            .collect()
    }

    /// A column holding both strings and integers must sort without panicking.
    ///
    /// `slice::sort_by` (driftsort) verifies its comparator's totality above
    /// the insertion-sort cutoff and panics with "user-provided comparison
    /// function does not correctly implement a total order" when it fails.
    /// `compare_sort_keys` used to *skip* incomparable pairs, so a
    /// string-vs-int pair reported `Equal` while int-vs-int pairs ordered —
    /// intransitive, and `MATCH (n:S) RETURN n.nm ORDER BY n.k DESC` over such
    /// a column aborted the query (a Python exception through pyo3; an
    /// unwinding panic with no `catch_unwind` in the Bolt server).
    #[test]
    fn mixed_type_column_sorts_without_panicking() {
        for n in [21usize, 24, 32, 64, 400] {
            let rows = mixed_rows(n);
            for specs in [vec![asc()], vec![desc()]] {
                let mut idx: Vec<usize> = (0..rows.len()).collect();
                idx.sort_by(|&a, &b| compare_sort_keys(&rows[a], &rows[b], &specs));
                assert_eq!(idx.len(), n);
            }
        }
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
