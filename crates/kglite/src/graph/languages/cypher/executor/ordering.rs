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
use std::collections::BinaryHeap;
use std::sync::Arc;

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

        if let Some(ordering) = crate::graph::core::filtering::compare_values(key_a, key_b) {
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

/// One retained candidate: its sort-key tuple, its input position (`seq`) and
/// the caller's payload. `specs` rides along because `BinaryHeap` orders
/// through `Ord`, which sees only the entries themselves.
struct Entry<P> {
    keys: Vec<Value>,
    seq: usize,
    payload: P,
    specs: Arc<[SortSpec]>,
}

impl<P> Entry<P> {
    /// Better-first rank. Ties on every key break by input position, so the
    /// retained set and its order match a *stable* full sort exactly.
    fn rank(&self, other: &Self) -> Ordering {
        compare_sort_keys(&self.keys, &other.keys, &self.specs)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

impl<P> PartialEq for Entry<P> {
    fn eq(&self, other: &Self) -> bool {
        self.rank(other) == Ordering::Equal
    }
}

impl<P> Eq for Entry<P> {}

impl<P> PartialOrd for Entry<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for Entry<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap and `rank` is better-first, so the root
        // holds the *worst* retained entry — exactly what overflow evicts.
        self.rank(other)
    }
}

/// Bounded top-K heap: retains the `limit` best entries under
/// [`compare_sort_keys`], in O(n log k) time and O(k) memory.
pub(crate) struct TopKCollector<P> {
    limit: usize,
    specs: Arc<[SortSpec]>,
    heap: BinaryHeap<Entry<P>>,
}

impl<P> TopKCollector<P> {
    pub(crate) fn new(specs: Vec<SortSpec>, limit: usize) -> Self {
        TopKCollector {
            limit,
            specs: specs.into(),
            heap: BinaryHeap::with_capacity(limit.saturating_add(1).min(1024)),
        }
    }

    /// Would `keys` at input position `seq` enter the current top-K?
    ///
    /// Purely an allocation guard — [`push`](Self::push) is correct on its own,
    /// evicting the worst entry on overflow. Callers evaluate sort keys into a
    /// reusable buffer and only pay for an owned tuple when this returns true,
    /// which is the difference between O(n) and O(k log n) key allocations.
    pub(crate) fn accepts(&self, keys: &[Value], seq: usize) -> bool {
        if self.limit == 0 {
            return false;
        }
        if self.heap.len() < self.limit {
            return true;
        }
        match self.heap.peek() {
            Some(root) => {
                compare_sort_keys(keys, &root.keys, &self.specs).then_with(|| seq.cmp(&root.seq))
                    == Ordering::Less
            }
            None => true,
        }
    }

    pub(crate) fn push(&mut self, keys: Vec<Value>, seq: usize, payload: P) {
        if self.limit == 0 {
            return;
        }
        self.heap.push(Entry {
            keys,
            seq,
            payload,
            specs: Arc::clone(&self.specs),
        });
        if self.heap.len() > self.limit {
            self.heap.pop();
        }
    }

    /// Drain into result order — best first — as `(sort keys, payload)` pairs.
    /// The keys come back so callers can reuse them for RETURN items that *are*
    /// the sort key instead of re-evaluating the expression.
    pub(crate) fn into_sorted(self) -> Vec<(Vec<Value>, P)> {
        self.heap
            .into_sorted_vec()
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
                collector.push(keys.clone(), seq, seq);
            }
        }
        collector
            .into_sorted()
            .into_iter()
            .map(|(_, payload)| payload)
            .collect()
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
