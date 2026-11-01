//! Stateless column kernel for ungrouped numeric scans of one node type.
//!
//! Walks the type's `Int64`/`Float64` column slices in row order, in chunks,
//! and produces a count and an exact integer sum over the rows that are live
//! (not tombstoned), non-null in every referenced column, and satisfy a
//! conjunction of [`NumericTest`]s. It keeps no state between calls.
//!
//! The kernel does not know which nodes the rows belong to. It reports
//! `live_rows_visited`; the caller compares that with the type's node count
//! and discards the answer on a mismatch, because a store row that is live but
//! owned by no node of the type (or two rows for one node) would otherwise be
//! counted silently.
//!
//! Same-kind pairings (Int column against Int literal, Float against Float)
//! compile to branch-free range compares. Mixed pairings keep the scalar
//! [`NumericTest`], the reference the differential test checks both against.

use super::numeric_cell_test::{Lit, NumericTest};

const CHUNK: usize = 1024;

/// One column's raw cells, with its null bytes (`0` = non-null).
#[derive(Clone, Copy)]
pub(super) enum Cells<'a> {
    Int { data: &'a [i64], nulls: &'a [u8] },
    Float { data: &'a [f64], nulls: &'a [u8] },
}

impl Cells<'_> {
    fn len(&self) -> usize {
        match self {
            Cells::Int { data, nulls } => data.len().min(nulls.len()),
            Cells::Float { data, nulls } => data.len().min(nulls.len()),
        }
    }
}

/// A predicate: one column against one test.
#[derive(Clone, Copy)]
pub(super) struct Pred<'a> {
    pub cells: Cells<'a>,
    pub test: NumericTest,
}

/// What the kernel found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KernelOut {
    /// Live rows satisfying every predicate.
    pub count: u64,
    /// Exact sum of the sum column over those rows (nulls skipped).
    pub sum: i128,
    /// Non-tombstoned rows walked, regardless of nulls or predicates.
    pub live_rows_visited: u64,
}

/// A predicate lowered to its fastest exact form.
enum Lowered<'a> {
    /// `lo <= cell <= hi`; `lo > hi` is the empty range.
    IntRange {
        data: &'a [i64],
        nulls: &'a [u8],
        lo: i64,
        hi: i64,
    },
    /// `cell > lo || (lo_inc && cell == lo)` and the upper twin; NaN fails.
    FloatRange {
        data: &'a [f64],
        nulls: &'a [u8],
        lo: f64,
        lo_inc: bool,
        hi: f64,
        hi_inc: bool,
    },
    ScalarInt {
        data: &'a [i64],
        nulls: &'a [u8],
        test: NumericTest,
    },
    ScalarFloat {
        data: &'a [f64],
        nulls: &'a [u8],
        test: NumericTest,
    },
}

const EMPTY_INT: (i64, i64) = (1, 0);

/// Inclusive integer bounds for `test` against an Int literal, or `None` when
/// the literal is a float.
fn int_bounds(test: NumericTest) -> Option<(i64, i64)> {
    let int = |l: Lit| match l {
        Lit::Int(t) => Some(t),
        Lit::Float(_) => None,
    };
    Some(match test {
        NumericTest::Equals(t) => {
            let t = int(t)?;
            (t, t)
        }
        NumericTest::Greater(t) => int(t)?.checked_add(1).map_or(EMPTY_INT, |l| (l, i64::MAX)),
        NumericTest::GreaterOrEqual(t) => (int(t)?, i64::MAX),
        NumericTest::Less(t) => int(t)?.checked_sub(1).map_or(EMPTY_INT, |h| (i64::MIN, h)),
        NumericTest::LessOrEqual(t) => (i64::MIN, int(t)?),
        NumericTest::Range {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } => {
            let (l, u) = (int(lower)?, int(upper)?);
            let lo = if lower_inclusive {
                Some(l)
            } else {
                l.checked_add(1)
            };
            let hi = if upper_inclusive {
                Some(u)
            } else {
                u.checked_sub(1)
            };
            match (lo, hi) {
                (Some(lo), Some(hi)) => (lo, hi),
                _ => EMPTY_INT,
            }
        }
    })
}

/// `(lo, lo_inclusive, hi, hi_inclusive)` for a Float literal, or `None`.
fn float_bounds(test: NumericTest) -> Option<(f64, bool, f64, bool)> {
    let fl = |l: Lit| match l {
        Lit::Float(t) => Some(t),
        Lit::Int(_) => None,
    };
    let (ni, pi) = (f64::NEG_INFINITY, f64::INFINITY);
    Some(match test {
        NumericTest::Equals(t) => (fl(t)?, true, fl(t)?, true),
        NumericTest::Greater(t) => (fl(t)?, false, pi, true),
        NumericTest::GreaterOrEqual(t) => (fl(t)?, true, pi, true),
        NumericTest::Less(t) => (ni, true, fl(t)?, false),
        NumericTest::LessOrEqual(t) => (ni, true, fl(t)?, true),
        NumericTest::Range {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } => (fl(lower)?, lower_inclusive, fl(upper)?, upper_inclusive),
    })
}

fn lower<'a>(p: &Pred<'a>) -> Lowered<'a> {
    match p.cells {
        Cells::Int { data, nulls } => match int_bounds(p.test) {
            Some((lo, hi)) => Lowered::IntRange {
                data,
                nulls,
                lo,
                hi,
            },
            None => Lowered::ScalarInt {
                data,
                nulls,
                test: p.test,
            },
        },
        Cells::Float { data, nulls } => match float_bounds(p.test) {
            Some((lo, lo_inc, hi, hi_inc)) => Lowered::FloatRange {
                data,
                nulls,
                lo,
                lo_inc,
                hi,
                hi_inc,
            },
            None => Lowered::ScalarFloat {
                data,
                nulls,
                test: p.test,
            },
        },
    }
}

impl Lowered<'_> {
    /// AND this predicate's verdict for rows `start..start + n` into `mask`.
    #[inline]
    fn apply(&self, start: usize, n: usize, mask: &mut [u8; CHUNK]) {
        let m = &mut mask[..n];
        let end = start + n;
        match *self {
            Lowered::IntRange {
                data,
                nulls,
                lo,
                hi,
            } => {
                for ((m, &x), &z) in m.iter_mut().zip(&data[start..end]).zip(&nulls[start..end]) {
                    *m &= ((z == 0) & (x >= lo) & (x <= hi)) as u8;
                }
            }
            Lowered::FloatRange {
                data,
                nulls,
                lo,
                lo_inc,
                hi,
                hi_inc,
            } => {
                for ((m, &x), &z) in m.iter_mut().zip(&data[start..end]).zip(&nulls[start..end]) {
                    let above = (x > lo) | (lo_inc & (x == lo));
                    let below = (x < hi) | (hi_inc & (x == hi));
                    *m &= ((z == 0) & above & below) as u8;
                }
            }
            Lowered::ScalarInt { data, nulls, test } => {
                for ((m, &x), &z) in m.iter_mut().zip(&data[start..end]).zip(&nulls[start..end]) {
                    *m &= ((z == 0) && test.holds_int(x)) as u8;
                }
            }
            Lowered::ScalarFloat { data, nulls, test } => {
                for ((m, &x), &z) in m.iter_mut().zip(&data[start..end]).zip(&nulls[start..end]) {
                    *m &= ((z == 0) && test.holds_float(x)) as u8;
                }
            }
        }
    }
}

/// Walk `rows` rows. `tombstones` may be shorter than `rows` (the rest are
/// live). Returns `None` when any referenced column holds fewer than `rows`
/// cells, because the kernel could not then answer for every row.
pub(super) fn scan(
    rows: usize,
    tombstones: &[bool],
    preds: &[Pred<'_>],
    sum_col: Option<(&[i64], &[u8])>,
) -> Option<KernelOut> {
    if preds.iter().any(|p| p.cells.len() < rows)
        || sum_col.is_some_and(|(d, n)| d.len() < rows || n.len() < rows)
    {
        return None;
    }
    let lowered: Vec<Lowered<'_>> = preds.iter().map(lower).collect();
    let (mut count, mut live, mut sum) = (0u64, 0u64, 0i128);
    let mut mask = [0u8; CHUNK];
    let mut start = 0;
    while start < rows {
        let n = CHUNK.min(rows - start);
        match tombstones.get(start..start + n) {
            Some(t) => {
                for (m, &dead) in mask[..n].iter_mut().zip(t) {
                    *m = !dead as u8;
                }
            }
            None => {
                for (i, m) in mask[..n].iter_mut().enumerate() {
                    *m = !tombstones.get(start + i).copied().unwrap_or(false) as u8;
                }
            }
        }
        live += mask[..n].iter().map(|&m| m as u64).sum::<u64>();
        for p in &lowered {
            p.apply(start, n, &mut mask);
        }
        count += mask[..n].iter().map(|&m| m as u64).sum::<u64>();
        if let Some((data, nulls)) = sum_col {
            // x = hi * 2^32 + lo with 0 <= lo < 2^32: both partial sums stay
            // far inside i64 for a 1024-row chunk, and the vectorised loop
            // needs no 128-bit lane.
            let (mut hi, mut lo) = (0i64, 0i64);
            for ((&m, &x), &z) in mask[..n]
                .iter()
                .zip(&data[start..start + n])
                .zip(&nulls[start..start + n])
            {
                let keep = -((m & (z == 0) as u8) as i64);
                let v = x & keep;
                hi += v >> 32;
                lo += v & 0xffff_ffff;
            }
            sum += ((hi as i128) << 32) + lo as i128;
        }
        start += n;
    }
    Some(KernelOut {
        count,
        sum,
        live_rows_visited: live,
    })
}

/// The caller's check that the walk covered exactly the type's nodes.
pub(super) fn cardinality_matches(out: &KernelOut, type_node_count: usize) -> bool {
    out.live_rows_visited == type_node_count as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_int(op: &str, t: i64) -> NumericTest {
        match op {
            ">" => NumericTest::Greater(Lit::Int(t)),
            ">=" => NumericTest::GreaterOrEqual(Lit::Int(t)),
            "<" => NumericTest::Less(Lit::Int(t)),
            "<=" => NumericTest::LessOrEqual(Lit::Int(t)),
            _ => NumericTest::Equals(Lit::Int(t)),
        }
    }

    /// Hand-computed: 8 rows, row 1 tombstoned, row 3 null year, row 6 null
    /// score. year > 2000 AND score < 0.5:
    ///   row0 2010/0.1 yes; row1 dead; row2 1990/0.2 no; row3 null; row4
    ///   2020/0.9 no; row5 2005/0.4 yes; row6 null score; row7 2001/0.0 yes.
    #[test]
    fn counts_and_sums_by_hand_with_tombstones_and_nulls() {
        let year = [2010i64, 2015, 1990, 0, 2020, 2005, 2002, 2001];
        let ynull = [0u8, 0, 0, 1, 0, 0, 0, 0];
        let score = [0.1f64, 0.1, 0.2, 0.3, 0.9, 0.4, 0.0, 0.0];
        let snull = [0u8, 0, 0, 0, 0, 0, 1, 0];
        let tomb = [false, true, false, false, false, false, false, false];
        let preds = [
            Pred {
                cells: Cells::Int {
                    data: &year,
                    nulls: &ynull,
                },
                test: test_int(">", 2000),
            },
            Pred {
                cells: Cells::Float {
                    data: &score,
                    nulls: &snull,
                },
                test: NumericTest::Less(Lit::Float(0.5)),
            },
        ];
        let out = scan(8, &tomb, &preds, Some((&year, &ynull))).unwrap();
        assert_eq!(out.count, 3);
        assert_eq!(out.sum, 2010 + 2005 + 2001);
        assert_eq!(out.live_rows_visited, 7);
        // The cardinality comparison is evaluated, and can fail.
        assert!(cardinality_matches(&out, 7));
        assert!(!cardinality_matches(&out, 8));
        // A shorter tombstone slice leaves the trailing rows live.
        let out = scan(8, &tomb[..2], &preds, None).unwrap();
        assert_eq!(out.live_rows_visited, 7);
        // A column shorter than the walk declines.
        assert!(scan(9, &tomb, &preds, None).is_none());
    }

    /// The lowered range forms agree with the scalar `NumericTest` on edge
    /// values, every operator, every pairing.
    #[test]
    fn lowered_forms_agree_with_the_scalar_reference() {
        let two53 = 1i64 << 53;
        let ints = [
            i64::MIN,
            i64::MIN + 1,
            -two53 - 1,
            -two53,
            -1,
            0,
            1,
            two53,
            two53 + 1,
            i64::MAX - 1,
            i64::MAX,
        ];
        let floats = [
            f64::NAN,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            0.5,
            two53 as f64,
            f64::INFINITY,
            -1.5,
        ];
        let lits: Vec<Lit> = ints
            .iter()
            .map(|&i| Lit::Int(i))
            .chain(floats.iter().map(|&f| Lit::Float(f)))
            .collect();
        let mut tests = Vec::new();
        for &l in &lits {
            tests.push(NumericTest::Equals(l));
            tests.push(NumericTest::Greater(l));
            tests.push(NumericTest::GreaterOrEqual(l));
            tests.push(NumericTest::Less(l));
            tests.push(NumericTest::LessOrEqual(l));
        }
        for &a in &lits {
            for &b in &lits {
                for (li, ui) in [(true, true), (false, false), (true, false)] {
                    tests.push(NumericTest::Range {
                        lower: a,
                        lower_inclusive: li,
                        upper: b,
                        upper_inclusive: ui,
                    });
                }
            }
        }
        let inulls = vec![0u8; ints.len()];
        let fnulls = vec![0u8; floats.len()];
        let none: [bool; 0] = [];
        for &t in &tests {
            let pi = [Pred {
                cells: Cells::Int {
                    data: &ints,
                    nulls: &inulls,
                },
                test: t,
            }];
            let want = ints.iter().filter(|&&x| t.holds_int(x)).count() as u64;
            assert_eq!(scan(ints.len(), &none, &pi, None).unwrap().count, want);
            let pf = [Pred {
                cells: Cells::Float {
                    data: &floats,
                    nulls: &fnulls,
                },
                test: t,
            }];
            let want = floats.iter().filter(|&&x| t.holds_float(x)).count() as u64;
            assert_eq!(scan(floats.len(), &none, &pf, None).unwrap().count, want);
        }
    }

    #[test]
    fn int_sum_is_exact_past_i64() {
        let data = [i64::MAX, i64::MAX, i64::MIN, 7];
        let nulls = [0u8; 4];
        let out = scan(4, &[], &[], Some((&data, &nulls))).unwrap();
        assert_eq!(
            out.sum,
            i64::MAX as i128 * 2 + i64::MIN as i128 + 7,
            "sum must not wrap"
        );
        assert_eq!(out.count, 4);
    }
}

/// Release-profile measurement of the kernel against the shipped fused path.
///
/// `cargo test -p kglite --release --lib -- --ignored --nocapture column_kernel_ab`
#[cfg(test)]
mod kernel_probe {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::schema::InternedKey;
    use crate::graph::session::{execute_read, ExecuteOptions};
    use crate::graph::storage::column_store::TypedColumn;
    use std::collections::HashMap;
    use std::time::Instant;

    const N: i64 = 1_000_000;

    fn paper_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        let columns: Vec<String> = ["id", "title", "year", "score"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let rows: Vec<Vec<Value>> = (0..N)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("Paper_{i}")),
                    Value::Int64(1950 + (next() % 76) as i64),
                    Value::Float64((next() >> 11) as f64 / (1u64 << 53) as f64),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            df,
            "Paper".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        graph.enable_columnar();
        graph
    }

    fn cypher_i64(graph: &DirGraph, q: &str) -> i64 {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let out = execute_read(graph, q, &opts).unwrap_or_else(|e| panic!("{q}: {e}"));
        match &out.result.rows[0][0] {
            Value::Int64(i) => *i,
            other => panic!("{q}: unexpected {other:?}"),
        }
    }

    fn min_ns(rounds: usize, mut f: impl FnMut()) -> f64 {
        (0..rounds)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e9
            })
            .fold(f64::MAX, f64::min)
    }

    #[test]
    #[ignore = "perf probe - release profile only"]
    fn column_kernel_ab() {
        if cfg!(debug_assertions) {
            panic!("run this probe with --release; a debug-profile number is invalid");
        }
        let graph = paper_graph();
        let store = graph
            .column_stores_by_name()
            .into_iter()
            .find(|(n, _)| *n == "Paper")
            .map(|(_, s)| s.clone())
            .expect("Paper store");
        assert!(store.columns_cover_rows() && !store.has_overflow());
        let col = |k: &str| {
            store
                .column(store.slot(InternedKey::from_str(k)).unwrap() as usize)
                .unwrap()
        };
        let (
            TypedColumn::Int64 {
                data: yd,
                nulls: yn,
            },
            TypedColumn::Float64 {
                data: sd,
                nulls: sn,
            },
        ) = (col("year"), col("score"))
        else {
            panic!("expected Int64 year, Float64 score");
        };
        let (yd, yn, sd, sn) = (yd.as_slice(), yn.as_slice(), sd.as_slice(), sn.as_slice());
        let rows = store.row_count() as usize;
        let tomb = store.base_tombstones();
        let preds = [
            Pred {
                cells: Cells::Int {
                    data: yd,
                    nulls: yn,
                },
                test: NumericTest::Greater(Lit::Int(2000)),
            },
            Pred {
                cells: Cells::Float {
                    data: sd,
                    nulls: sn,
                },
                test: NumericTest::Less(Lit::Float(0.5)),
            },
        ];
        let type_count = graph.type_indices.get("Paper").unwrap().len();
        assert_eq!(rows, N as usize);

        let k_count = scan(rows, tomb, &preds, None).unwrap();
        let k_sum = scan(rows, tomb, &[], Some((yd, yn))).unwrap();
        assert!(
            cardinality_matches(&k_count, type_count),
            "{k_count:?} vs {type_count}"
        );
        assert!(cardinality_matches(&k_sum, type_count));

        let q_count = "MATCH (n:Paper) WHERE n.year > 2000 AND n.score < 0.5 RETURN count(n) AS c";
        let q_sum = "MATCH (n:Paper) RETURN sum(n.year) AS s";
        let e_count = cypher_i64(&graph, q_count);
        let e_sum = cypher_i64(&graph, q_sum);
        assert_eq!(k_count.count as i64, e_count, "count equality");
        assert_eq!(k_sum.sum, e_sum as i128, "sum equality");
        println!(
            "answers: count={e_count} sum={e_sum} live={}",
            k_count.live_rows_visited
        );

        let rounds = 25;
        let per_row = |ns: f64| ns / rows as f64;
        let kc = min_ns(rounds, || {
            std::hint::black_box(scan(rows, tomb, std::hint::black_box(&preds), None));
        });
        let ks = min_ns(rounds, || {
            std::hint::black_box(scan(rows, tomb, &[], Some((yd, yn))));
        });
        let ec = min_ns(rounds, || {
            std::hint::black_box(cypher_i64(&graph, q_count));
        });
        let es = min_ns(rounds, || {
            std::hint::black_box(cypher_i64(&graph, q_sum));
        });
        println!("{:<28} {:>10} {:>10}", "cell (min of 25)", "ms", "ns/row");
        for (l, ns) in [
            ("kernel count 2-pred", kc),
            ("kernel sum(year)", ks),
            ("existing count 2-pred", ec),
            ("existing sum(year)", es),
        ] {
            println!("{l:<28} {:>10.3} {:>10.2}", ns / 1e6, per_row(ns));
        }
        drop(graph);
    }
}
