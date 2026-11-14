//! What each structural check of the eager verify tier refuses, and that the
//! default lazy open reads the same files without objecting. Every file is
//! CRC-valid (the section and metadata digests are re-sealed), so only the
//! structural check can object.

use super::container_test_support::{patch_section, read_section};
use super::frozen_tests::saved_topo;
use super::index_sections_tests::rows;
use super::*;

fn open(path: &std::path::Path, verify: Verify) -> io::Result<Arc<DirGraph>> {
    load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new()
            .with_storage(StorageMode::Mapped)
            .with_verify(verify),
    )
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn put_u32(bytes: &mut [u8], index: usize, value: u32) {
    bytes[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
}

/// The first node whose outgoing run holds two different type ordinals:
/// `(run start, node)`.
fn mixed_run(path: &std::path::Path) -> usize {
    let off = u32s(&read_section(path, "topo:out_off"));
    let tys: Vec<u16> = read_section(path, "topo:out_ty")
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    off.windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .find(|&(lo, hi)| tys[lo..hi].windows(2).any(|p| p[0] != p[1]))
        .expect("the hub has edges of several types")
        .0
}

type Craft = fn(&std::path::Path);

/// `(case, section named by the refusal, whether a lazy open answers
/// differently from the honest file, craft)`.
const CASES: &[(&str, &str, bool, Craft)] = &[
    (
        "endpoint src past the node array",
        "topo:edges",
        false,
        |p| {
            patch_section(p, "topo:edges", |b| {
                let live = b
                    .chunks_exact_mut(12)
                    .find(|r| r[..4] != [0xFF; 4])
                    .unwrap();
                live[..4].copy_from_slice(&900_000u32.to_le_bytes());
            })
        },
    ),
    (
        "endpoint type ordinal past the list",
        "topo:edges",
        false,
        |p| {
            patch_section(p, "topo:edges", |b| {
                let live = b
                    .chunks_exact_mut(12)
                    .find(|r| r[..4] != [0xFF; 4])
                    .unwrap();
                live[8..10].copy_from_slice(&9999u16.to_le_bytes());
            })
        },
    ),
    ("adjacency peer changed", "topo:out_edges", true, |p| {
        patch_section(p, "topo:out_edges", |b| put_u32(b, 0, 900_000))
    }),
    (
        "adjacency edge slot past the array",
        "topo:out_edges",
        false,
        |p| patch_section(p, "topo:out_edges", |b| put_u32(b, 1, 0xFFFF_FF00)),
    ),
    (
        "adjacency lists one edge twice",
        "topo:in_edges",
        false,
        |p| {
            patch_section(p, "topo:in_edges", |b| {
                let second = u32s(b)[3];
                put_u32(b, 1, second);
            })
        },
    ),
    (
        "adjacency type ordinal past the list",
        "topo:out_edges",
        true,
        |p| {
            patch_section(p, "topo:out_ty", |b| {
                b[..2].copy_from_slice(&9999u16.to_le_bytes())
            })
        },
    ),
    ("typed run out of order", "topo:out_ty", true, |p| {
        let at = mixed_run(p);
        patch_section(p, "topo:out_ty", |b| {
            let (x, y) = (
                b[2 * at..2 * at + 2].to_vec(),
                b[2 * at + 2..2 * at + 4].to_vec(),
            );
            b[2 * at..2 * at + 2].copy_from_slice(&y);
            b[2 * at + 2..2 * at + 4].copy_from_slice(&x);
        });
        patch_section(p, "topo:out_edges", |b| {
            let (x, y) = (
                u32s(b)[2 * at..2 * at + 2].to_vec(),
                u32s(b)[2 * at + 2..2 * at + 4].to_vec(),
            );
            put_u32(b, 2 * at, y[0]);
            put_u32(b, 2 * at + 1, y[1]);
            put_u32(b, 2 * at + 2, x[0]);
            put_u32(b, 2 * at + 3, x[1]);
        });
    }),
    ("type mask cleared", "topo:type_mask", true, |p| {
        patch_section(p, "topo:type_mask", |b| b.fill(0))
    }),
    ("peer count raised", "idx:peer:K1:out", true, |p| {
        for part in ["out", "in"] {
            patch_section(p, &format!("idx:peer:K1:{part}"), |b| {
                put_u32(b, 1, u32s(b)[1] + 1000)
            });
        }
    }),
    ("id index nodes swapped", "idx:id:A", true, |p| {
        patch_section(p, "idx:id:A:nodes", |b| {
            let v = u32s(b);
            put_u32(b, 0, v[1]);
            put_u32(b, 1, v[0]);
        })
    }),
];

const PROBES: &[&str] = &[
    "MATCH (a)-[r:K1]->(b) RETURN count(r) AS c, sum(b.id) AS s",
    "MATCH (a:A)-[:K1]->(b:B) RETURN a.id AS a, count(b) AS c ORDER BY a",
    "MATCH (h:H)-[r]->(x) RETURN type(r) AS t, count(x) AS c ORDER BY t",
    "MATCH (a:A)-[:K1|K2]->(x) RETURN a.id AS a, collect(x.id) AS xs ORDER BY a",
    "MATCH (b:B)<-[:K1]-(a) RETURN b.id AS b, count(a) AS c ORDER BY b",
    "MATCH (a)-[r]->(b) RETURN a.id AS a, count(b) AS c, sum(b.id) AS s ORDER BY a LIMIT 40",
    "MATCH (a)-[:K1]->(b) RETURN b.id AS b, count(*) AS c ORDER BY c DESC, b LIMIT 5",
    "MATCH (n:A {id: 1}) RETURN n.n AS n",
];

/// Every probe's rows, as one string.
fn answers(graph: &DirGraph) -> String {
    PROBES
        .iter()
        .map(|q| rows(graph, q))
        .collect::<Vec<_>>()
        .join("|")
}

/// What a lazy open, `verify_graph` and an eager open do with one crafted
/// file, as the first thing that is not as expected.
fn judge(section: &str, path: &std::path::Path) -> Result<(), String> {
    let lazy = open(path, Verify::Lazy).map_err(|e| format!("a lazy open objected: {e}"))?;
    let _ = answers(&lazy);
    let report = verify_graph(&lazy);
    let failed = report.sections.iter().find(|s| s.error.is_some());
    if !failed.is_some_and(|s| s.name.starts_with(section)) {
        return Err(format!("verify_graph names {failed:?}"));
    }
    let error = open(path, Verify::Eager)
        .err()
        .ok_or("an eager open did not refuse")?;
    if error.kind() != io::ErrorKind::InvalidData || !error.to_string().contains(section) {
        return Err(format!("the eager refusal is {error}"));
    }
    Ok(())
}

#[test]
fn eager_refuses_each_structural_inconsistency_and_lazy_opens_it() {
    let problems: Vec<String> = CASES
        .iter()
        .filter_map(|(case, section, _, craft)| {
            let (_tmp, path) = saved_topo();
            craft(&path);
            judge(section, &path).err().map(|e| format!("{case}: {e}"))
        })
        .collect();
    assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn an_honest_file_passes_the_eager_tier() {
    let (_tmp, path) = saved_topo();
    let eager = open(&path, Verify::Eager).expect("an honest file opens eagerly");
    assert!(verify_graph(&eager).is_ok());
}

/// Under the lazy open a file that fails a check answers, and for each of
/// these the answer is not the honest one: the contract is pinned both ways.
#[test]
fn a_lazy_open_pins_the_wrong_answers_the_eager_tier_exists_for() {
    let (_tmp, path) = saved_topo();
    let honest = answers(&open(&path, Verify::Lazy).unwrap());
    for (case, _, changes, craft) in CASES {
        let (_tmp, path) = saved_topo();
        craft(&path);
        let crafted = answers(&open(&path, Verify::Lazy).unwrap());
        assert_eq!(crafted != honest, *changes, "{case}");
    }
}
